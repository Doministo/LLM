import streamlit as st
import requests
import json
from datetime import datetime
import wikipedia
import pandas as pd
import docx
import fitz  # from PyMuPDF
import io
from bs4 import BeautifulSoup
import traceback
import logging
import re
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pygments import highlight
import os

# Configure logging - musí být na začátku
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(
    page_title="LLM Chat – Ollama",
    page_icon="🤖",
    layout="wide",
    menu_items={'About': "### Lokální LLM chat pomocí Ollama a Streamlit"}
)

# CSS z externího souboru
def load_css(file_name):
    # o úroveň výš vůči aktuálnímu skriptu
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", file_name))
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ CSS soubor `{file_name}` nebyl nalezen. Stylování nemusí fungovat správně.")

load_css("style.css")

# --- Konfigurace ---
ALLOWED_FILE_TYPES = ["txt", "pdf", "docx", "csv", "py", "html", "css", "json", "md"]
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_VERSION_URL = "/api/version"
OLLAMA_TAGS_URL = "/api/tags"
OLLAMA_GENERATE_URL = "/api/generate"
TOKENIZER_NAME = "gpt2"

try:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
    logging.info("Tokenizer byl úspěšně inicializován.")
except ImportError:
    logging.warning("Balíček transformers není nainstalován. Používá se fallback metoda.")
    tokenizer = None

def count_tokens(text: str) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    # Fallback: jednoduché dělení textu na mezery
    return len(text.split())

# --- Pomocné funkce ---
def safe_get_digest(model_data: dict) -> str:
    try:
        digest = model_data.get('digest', '')
        if digest and ':' in digest:
            return digest.split(':')[-1][:6]
        return digest[:6] if digest else "latest"
    except Exception as e:
        logging.warning(f"Could not parse digest from model_data: {model_data}. Error: {e}")
        return "unknown"

def load_available_models(host: str) -> list:
    try:
        url = f"{host.strip('/')}{OLLAMA_TAGS_URL}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        models_data = response.json().get('models', [])
        formatted_models = [
            f"{model.get('name', 'unknown_model')}:{safe_get_digest(model)}"
            for model in models_data
        ]
        logging.info(f"Loaded models from {host}: {formatted_models}")
        return formatted_models
    except requests.exceptions.RequestException as e:
        st.error(f"Chyba při připojení k Ollama API ({host}): {e}")
        logging.error(f"Error connecting to Ollama API ({host}): {e}")
        return []
    except json.JSONDecodeError as e:
        st.error("Chyba při čtení odpovědi od Ollama (neplatný JSON).")
        logging.error(f"Error decoding JSON response from Ollama: {e}")
        return []
    except Exception as e:
        st.error(f"Neočekávaná chyba při načítání modelů: {e}")
        logging.error(f"Unexpected error loading models: {e}")
        return []

def format_thinking_block(text):
    def replacer(match):
        full_tag = match.group(0)
        attrs = match.group(1)
        content = match.group(2).strip()

        time_match = re.search(r'time=["\']?([\d.]+)["\']?', attrs)
        time_text = f"{float(time_match.group(1)):.1f} sekundy" if time_match else "?"

        return f'''
        <details class="think-box">
            <summary>💭 Přemýšlení modelu (trvalo {time_text}) – klikni pro zobrazení</summary>
            <div class="think-content">{content}</div>
        </details>
        '''
    return re.sub(r'<think(.*?)>(.*?)</think>', replacer, text, flags=re.DOTALL)

# 🧠 Funkce pro shrnutí starší historie
def summarize_old_history(chat_history, model_name, host_url, max_tokens=512):
    if len(chat_history) <= 5:
        return None

    # Limit the number of messages to process for performance
    max_messages_to_process = 50
    limited_history = chat_history[-max_messages_to_process:-5] if len(chat_history) > max_messages_to_process else chat_history[:-5]

    to_summarize = ""
    for sender, message, timestamp in limited_history:
        role = "Uživatel" if sender == "user" else "Asistent"
        content = message.split("\n\n*Přibližný počet tokenů:\n")[0].strip()
        to_summarize += f"{timestamp} – {role}: {content}\n"

    prompt = f"Shrň důležité informace z následující konverzace:\n\n{to_summarize}"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "options": {
            "temperature": 0.5,
            "num_predict": max_tokens
        },
        "stream": False
    }

    try:
        response = requests.post(f"{host_url}/api/generate", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        logging.warning(f"Chyba při sumarizaci: {e}")
    return None


# 🧠 Funkce pro získání posledních zpráv do tokenového limitu
def get_recent_messages(chat_history, token_limit=1000):
    messages = []
    total_tokens = 0

    for sender, message, _ in reversed(chat_history):  # od konce
        content = message.split("\n\n*Přibližný počet tokenů:")[0].strip()
        token_count = count_tokens(content)
        if total_tokens + token_count > token_limit:
            break
        role = "user" if sender == "user" else "assistant"
        messages.insert(0, {"role": role, "content": content})
        total_tokens += token_count

    return messages


# 🧠 Funkce pro sestavení celého kontextu
def build_context_prompt(memory_summary, recent_messages, user_input, force_czech=False):
    """
    Builds the context prompt for the chat model.

    Args:
        memory_summary (str): A summary of previous chat history, if available.
        recent_messages (list): A list of recent messages to include in the context.
        user_input (str): The current user input to be added to the context.
        force_czech (bool): If True, forces the model to respond exclusively in Czech.

    Returns:
        list: A list of dictionaries representing the context for the chat model.
    """
    context = []
    if memory_summary:
        context.append({"role": "system", "content": memory_summary})
    if force_czech:
        context.append({"role": "system", "content": "Odpovídej výhradně v českém jazyce."})
    context.extend(recent_messages)
    context.append({"role": "user", "content": user_input.strip()})
    return context


# --- Inicializace stavu ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = st.session_state.get('model', "llama3")  # ""
if "system_info" not in st.session_state:
    st.session_state.system_info = ""
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = DEFAULT_OLLAMA_HOST

# --- Sidebar ---
with st.sidebar:
    # Načíst modely při startu stránky
    if "available_models" not in st.session_state:
        st.session_state["available_models"] = load_available_models(st.session_state.ollama_host)

    st.subheader("⚙️ Nastavení")

    st.session_state.ollama_host = st.text_input(
        "🔗 Ollama API adresa",
        value=st.session_state.get("ollama_host", DEFAULT_OLLAMA_HOST),  # Použij konstantu
        help="Adresa běžící Ollama instance (např. http://localhost:11434)"
    )

    # Načtení a výběr modelu
    models = st.session_state.get("available_models", [])
    if not models:
        st.warning("⚠️ Nelze načíst modely z Ollama.")
        models = []

    models_display = ["Zvolte model"] + models
    current_selection = st.session_state.get("selected_model", "Zvolte model")

    if current_selection not in models_display:
        st.session_state.selected_model = "Zvolte model"
        current_selection = "Zvolte model"
        logging.info(f"Model '{st.session_state.get('selected_model', 'N/A')}' not found in available models, resetting selection.")

    try:
        default_index = models_display.index(current_selection)
    except ValueError:
        default_index = 0  # Fallback na "Zvolte model"
        logging.warning(f"Could not find index for '{current_selection}' in {models_display}, defaulting to 0.")

    selected_model_value = st.selectbox(
        "🤖 Vyber model",
        models_display,
        index=default_index,
        key="selected_model"
    )

    temperature = st.slider("🌡️ Teplota", 0.0, 1.0, st.session_state.get("temperature", 0.8), 0.01, key="temperature")
    max_tokens = st.slider("🔢 Max. tokenů v odpovědi", 50, 4096, st.session_state.get("max_tokens", 512), 50, key="max_tokens")
    top_p = st.slider("📊 Top-p", 0.0, 1.0, st.session_state.get("top_p", 0.95), 0.01, key="top_p")
    use_memory = st.checkbox("🧠 Zapnout paměť", value=st.session_state.get("use_memory", False), key="use_memory")
    cz_mode = st.checkbox("🇨🇿 Vynutit češtinu", value=st.session_state.get("cz_mode", False), key="cz_mode")
    stream_mode = st.checkbox("🌀 Streamování odpovědí", value=st.session_state.get("stream_mode", True), key="stream_mode")
    show_tokens = st.checkbox("🔢 Zobrazovat tokeny", value=st.session_state.get("show_tokens", True), key="show_tokens")

# --- Hlavní část ---
st.title("🤖 Lokální LLM Chat")
st.caption(f"Powered by Ollama & Streamlit | Model: {st.session_state.selected_model or 'Není vybrán'}")

st.markdown('<div id="chat-container" class="chat-container">', unsafe_allow_html=True)
if st.session_state.chat_history:
    for sender, message, timestamp in (st.session_state.chat_history):
        css_class = "user-message" if sender == "user" else "bot-message"
        icon = '🧑' if sender == 'user' else '🤖'
        sender_name = 'Ty' if sender == 'user' else 'LLM'
        formatted_message = f"""
        <div class="{css_class}">
            <strong>{icon} {sender_name}</strong>
            <div style="margin-top: 0.3rem;">{format_thinking_block(message)}</div>
            <div style="font-size: 0.8rem; color: #888; text-align: right; margin-top: 0.5rem;">{timestamp}</div>
        </div>
        """
        st.markdown(formatted_message, unsafe_allow_html=True)
        st.write("")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="fixed-input">', unsafe_allow_html=True)

# --- PLUGINY: Wikipedia, Soubory, Web ---
wiki_context = ""
file_context = ""
web_context = ""
include_wiki = False
include_file = False
include_web = False

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📖 Wikipedia", "📂 Soubor", "🌍 Web"])

# 🧭 Wikipedia Plugin
with tab2:
    st.subheader("📖 Wikipedia vyhledávání")
    search_term = st.text_input("🔍 Heslo pro vyhledání", placeholder="např. Umělá inteligence", key="wiki_search")
    if search_term:
        try:
            wikipedia.set_lang("cs")
            search_results = wikipedia.search(search_term)
            if not search_results:
                st.error(f"❌ Žádné výsledky pro '{search_term}'.")
            else:
                try:
                    summary = wikipedia.summary(search_results[0], sentences=3, auto_suggest=True)
                except wikipedia.exceptions.DisambiguationError as e:
                    st.warning(f"🔀 '{search_term}' má více významů. Možnosti: {e.options[:5]}")
                except wikipedia.exceptions.PageError:
                    st.error(f"❌ Stránka '{search_results[0]}' nebyla nalezena.")
            st.markdown("#### Shrnutí článku")
            st.markdown(f"""
                <div style="font-size: 0.85rem; line-height: 1.4; color: #cccccc; background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem;">
                    {summary}
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            include_wiki = st.checkbox("Přidat shrnutí do dotazu", key="wiki_include")
            if include_wiki:
                wiki_context = summary
        except wikipedia.exceptions.PageError:
            st.error(f"❌ Stránka '{search_term}' nebyla nalezena.")
        except wikipedia.exceptions.DisambiguationError as e:
            st.warning(f"Vyber soubor ({', '.join(ALLOWED_FILE_TYPES).upper()})")
        except Exception as e:
            st.error(f"❌ Neočekávaná chyba při hledání na Wikipedii: {e}")
            logging.error(f"Wikipedia search error for '{search_term}': {e}")

# 📁 File Upload Plugin
with tab3:
    st.subheader("📂 Nahrát soubor pro kontext")
    uploaded_file = st.file_uploader(
        "Vyber soubor (TXT, PDF, DOCX, CSV, PY, HTML, CSS, JSON, MD)",
        type=["txt", "pdf", "docx", "csv", "py", "html", "css", "json", "md"],
        key="file_uploader"
    )
    if uploaded_file is not None:
        try:
            file_type = uploaded_file.name.split(".")[-1].lower()
            bytes_data = uploaded_file.getvalue()

            if file_type in ["txt"]:
                try:
                    raw_text = bytes_data.decode("utf-8")
                except UnicodeDecodeError:
                    raw_text = bytes_data.decode("latin-1")
                
                # Detekce a zvýraznění syntaxe
                file_context = raw_text

            elif file_type in ["py", "html", "css", "json", "md"]:
                try:
                    raw_text = bytes_data.decode("utf-8")
                except UnicodeDecodeError:
                    raw_text = bytes_data.decode("latin-1")
                
                try:
                    if file_type in ["py", "html", "css", "json", "md"]:
                        lexer = get_lexer_by_name(file_type)
                        formatter = HtmlFormatter(style="monokai", full=False, noclasses=True)
                        file_context = highlight(raw_text, lexer, formatter)
                    else:
                        st.warning(f"⚠️ Typ souboru '{file_type}' není podporován pro zvýraznění syntaxe.")
                        file_context = raw_text
                except Exception as e:
                    st.error(f"❌ Chyba při zvýrazňování syntaxe: {e}")
                    file_context = raw_text
                    file_context = raw_text

            elif file_type == "pdf":
                with fitz.open(stream=bytes_data, filetype="pdf") as doc:
                    file_context = "\n".join([page.get_text() for page in doc])

            elif file_type == "docx":
                doc = docx.Document(io.BytesIO(bytes_data))
                file_context = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

            elif file_type == "csv":
                df = pd.read_csv(io.BytesIO(bytes_data))
                file_context = df.to_string()

            if file_context.strip():
                if file_type not in ["py", "html", "css", "json", "md"]:
                    st.markdown("#### Náhled souboru", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='background-color:#1e1e1e; padding:1rem; border-radius:0.5rem; font-size:0.85rem;'>
                        {file_context}
                    </div>
                    """, unsafe_allow_html=True)
                include_file = st.checkbox("Přidat obsah souboru do dotazu", key="file_include")
            else:
                st.warning("⚠️ Obsah souboru je prázdný nebo jej nelze extrahovat.")

        except Exception as e:
            st.error(f"❌ Chyba při zpracování souboru '{uploaded_file.name}': {e}")
            logging.error(f"Error processing file '{uploaded_file.name}': {e}\n{traceback.format_exc()}")

# 🌐 Web Scraping Plugin
with tab4:
    st.subheader("🌍 Načíst textový obsah z webové stránky")
    web_url = st.text_input("🔗 Zadej URL adresu", placeholder="https://...", key="web_url")
    if web_url:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            page = requests.get(web_url, timeout=15, headers=headers)
            if 'text/html' in page.headers.get('Content-Type', ''):
                soup = BeautifulSoup(page.content, "html.parser")
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'article', 'section'], limit=100)
                extracted_text = "\n".join(t.get_text(separator=' ', strip=True) for t in texts if t.get_text(strip=True))
                if extracted_text.strip():
                    include_web = st.checkbox("Přidat text z webu do dotazu", key="web_include")
                    if include_web:
                        web_context = extracted_text
                else:
                    st.warning("⚠️ Nepodařilo se extrahovat čitelný text z hlavních tagů stránky.")
            else:
                st.warning(f"⚠️ Obsah na URL není HTML (Content-Type: {page.headers.get('Content-Type', 'N/A')}).")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Chyba při načítání URL: {e}")
            logging.error(f"Error fetching URL '{web_url}': {e}")
        except Exception as e:
            st.error(f"❌ Neočekávaná chyba při zpracování webu: {e}")
            logging.error(f"Error processing web content from '{web_url}': {e}\n{traceback.format_exc()}")

# --- Zpracování Chatu ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("💬 Napiš dotaz:", height=100, placeholder="Zadej svůj dotaz nebo instrukci...", key="user_input")
    submitted = st.form_submit_button("🚀 Odeslat")

st.markdown('</div>', unsafe_allow_html=True)

if submitted and user_input.strip() and st.session_state.selected_model and "Nebyly nalezeny modely" not in st.session_state.selected_model:
    
    # Sestavení historie pro kontext
    context_history = []
    
    selected_model_full = st.session_state.selected_model
    parts = selected_model_full.split(':')
    model_name_for_api = f"{parts[0]}:{parts[1]}" if len(parts) > 2 else selected_model_full

    # Shrnutí pokud je historie dlouhá
    if len(st.session_state.chat_history) > 10:
        summary = summarize_old_history(
            st.session_state.chat_history,
            model_name_for_api,
            st.session_state.ollama_host
        )
        if summary:
            st.session_state.memory_summary = f"🧠 Shrnutí předchozí konverzace:\n{summary}"
    
    # Poslední zprávy do limitu
    recent = get_recent_messages(st.session_state.chat_history)
    st.session_state.recent_memory = recent
    
    # Kontext pro prompt
    context_history = build_context_prompt(
        st.session_state.get("memory_summary", ""),
        recent,
        user_input,
        cz_mode
    )
    
    final_prompt = user_input.strip()
    prepended_context = ""
    if include_wiki and wiki_context:
        prepended_context += f"[WIKIPEDIA SHRNUTÍ]:\n{wiki_context}\n\n"
    if include_file and file_context:
        prepended_context += f"[OBSAH SOUBORU]:\n{file_context[:10000]}\n\n"
    if include_web and web_context:
        prepended_context += f"[TEXT Z WEBU]:\n{web_context[:10000]}\n\n"
    if prepended_context:
        final_prompt = f"{prepended_context}Můj dotaz s využitím předchozího kontextu:\n{final_prompt}"
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append(("user", user_input.strip(), timestamp))
    
    # Příprava payloadu pro Ollama API
    
    if not model_name_for_api or "Nebyly nalezeny modely" in model_name_for_api:
        st.warning("⚠️ Prosím, vyberte platný model v nastavení.")
    else:
        logging.info(f"Selected model: {selected_model_full}, API: {model_name_for_api}")
        final_answer = ""
        context_history.append({"role": "user", "content": final_prompt})
        payload = {
            "model": model_name_for_api,
            "messages": context_history,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": stream_mode
        }
        try:
            with st.spinner("🧠 Přemýšlím..."):
                response = requests.post(f"{st.session_state.ollama_host}{OLLAMA_GENERATE_URL}",
                                         json=payload, stream=stream_mode, timeout=120)
                if response.status_code == 200:
                    if stream_mode:
                        placeholder = st.empty()
                        for line in response.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line.decode())
                                    token = data.get("response", "")
                                    final_answer += token
                                    placeholder.markdown(f"**🤖 LLM:** {final_answer}")
                                except Exception as e:
                                    final_answer += f"\n[Chyba: {str(e)}]"
                    else:
                        data = response.json()
                        final_answer = data.get("response", "")
        except Exception as e:
            st.warning("⚠️ Nastala chyba při komunikaci s modelem.")
        
        # FALLBACK logika
        if not final_answer:
            # 🧠 Sestavení fallback promptu jako čistý text (bez messages)
            fallback_prompt_parts = []
            
            # 1. Shrnutí z paměti (dlouhodobá paměť)
            memory_summary = st.session_state.get("memory_summary", "").strip()
            if memory_summary:
                fallback_prompt_parts.append("🧠 Shrnutí předchozí konverzace:\n" + memory_summary)
            
            # 2. Instrukce o češtině
            if cz_mode:
                fallback_prompt_parts.append("🗣️ Odpovídej výhradně v českém jazyce. Nemusíš zmiňovat, že odpovídáš česky.")
            
            # 3. Poslední zprávy jako prostý text
            if 'recent' in locals() and recent:
                recent_lines = []
                for msg in recent:
                    role = "Uživatel" if msg["role"] == "user" else "Asistent"
                    content = msg["content"].strip()
                    recent_lines.append(f"{role}: {content}")
                fallback_prompt_parts.append("🕑 Poslední zprávy:\n" + "\n".join(recent_lines))
            
            # 4. Pluginové kontexty (wiki, soubor, web)
            if prepended_context.strip():
                fallback_prompt_parts.append("📎 Kontext:\n" + prepended_context.strip())
            
            # 5. Aktuální vstup uživatele
            fallback_prompt_parts.append("💬 Uživatelský dotaz:\n" + user_input.strip())
            
            # Finální fallback prompt sestavený ze všech částí
            fallback_prompt = "\n\n".join(fallback_prompt_parts)
            
            payload = {
                "model": model_name_for_api,
                "prompt": fallback_prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": stream_mode
            }
            try:
                with st.spinner("🧠 Přepínám na nouzový režim..."):
                    response = requests.post(f"{st.session_state.ollama_host}{OLLAMA_GENERATE_URL}",
                                             json=payload, stream=stream_mode, timeout=120)
                    if response.status_code == 200:
                        if stream_mode:
                            placeholder = st.empty()
                            for line in response.iter_lines():
                                if line:
                                    try:
                                        data = json.loads(line.decode())
                                        token = data.get("response", "")
                                        final_answer += token
                                        placeholder.markdown(f"**🤖 LLM:** {final_answer}")
                                    except Exception as e:
                                        final_answer += f"\n[Chyba: {str(e)}]"
                        else:
                            data = response.json()
                            final_answer = data.get("response", "")
                        if not final_answer:
                            st.warning("⚠️ Ani fallback nevrátil odpověď.")
            except requests.exceptions.Timeout:
                st.error("❌ Vypršel časový limit při čekání na odpověď od Ollama.")
                logging.error("Ollama API request timed out.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Chyba komunikace s Ollama API: {e}")
                logging.error(f"Ollama API request error: {e}\n{traceback.format_exc()}")
            except Exception as e:
                st.error(f"⚠️ Neočekávaná chyba při generování odpovědi: {e}")
                logging.error(f"Unexpected error during Ollama call: {e}\n{traceback.format_exc()}")
        
        if final_answer:
            token_count = count_tokens(final_answer)
            if show_tokens:
                final_answer += f"\n\n*Přibližný počet tokenů: {token_count}*"
            st.session_state.chat_history.append(("bot", final_answer, datetime.now().strftime("%H:%M:%S")))
            st.rerun()
        else:
            st.warning("⚠️ Model vrátil prázdnou odpověď.")
elif submitted and not user_input.strip():
    st.warning("⚠️ Prosím, zadejte dotaz.")
elif submitted and (not st.session_state.selected_model or "Nebyly nalezeny modely" in st.session_state.selected_model):
    st.warning("⚠️ Prosím, vyberte platný model v nastavení.")



if not st.session_state.system_info:
    try:
        version_url = f"{st.session_state.ollama_host.strip('/')}{OLLAMA_VERSION_URL}"
        system_response = requests.get(version_url, timeout=5)
        if system_response.status_code == 200:
            version_info = system_response.json()
            st.session_state.system_info = f"Ollama verze: {version_info.get('version', 'N/A')}"
            logging.info(f"Ollama version check successful: {st.session_state.system_info}")
            st.sidebar.caption(st.session_state.system_info)
        else:
            logging.warning(f"Ollama version check failed with status: {system_response.status_code}")
            st.sidebar.caption("⚠️ Nelze ověřit verzi Ollama.")
    except Exception as e:
        logging.warning(f"Failed to get Ollama version info: {e}")
        st.sidebar.caption("⚠️ Nelze se připojit k Ollama pro info.")