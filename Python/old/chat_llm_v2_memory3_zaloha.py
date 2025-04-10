import streamlit as st
import requests
import json
from datetime import datetime
from transformers import AutoTokenizer
import wikipedia
import pandas as pd
import docx
import fitz  # z PyMuPDF
import io
from bs4 import BeautifulSoup
import traceback
import logging
import re

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
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Funkce pro vytvoření shrnutí z historie chatu (kombinovaný styl)
def generate_combined_memory(chat_history):
    if not chat_history:
        return ""
    diary_part = "🧠 Kontext z minulých událostí:\n\n"
    bullet_part = "\n\n**• Klíčové body:**\n"
    for sender, message, timestamp in chat_history:
        role = "Uživatel" if sender == "user" else "Asistent"
        clean = message.split("\n\n*Přibližný počet tokenů:")[0].strip()
        diary_part += f"{timestamp} – {role} řekl: {clean}\n"
    for sender, message, _ in chat_history[-5:]:
        role = "Uživatel" if sender == "user" else "Asistent"
        clean = message.split("\n\n*Přibližný počet tokenů:")[0].strip()
        bullet_part += f"- {role}: {clean[:100]}{'...' if len(clean) > 100 else ''}\n"
    return diary_part + bullet_part

# --- Konfigurace ---
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_VERSION_URL = "/api/version"
OLLAMA_TAGS_URL = "/api/tags"
OLLAMA_GENERATE_URL = "/api/generate"
TOKENIZER_NAME = "gpt2"

# --- Inicializace tokenizeru ---
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    logging.info(f"Tokenizer '{TOKENIZER_NAME}' loaded successfully.")
except Exception as e:
    st.error(f"Chyba při načítání tokenizeru '{TOKENIZER_NAME}': {e}")
    logging.error(f"Failed to load tokenizer '{TOKENIZER_NAME}': {e}")

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



# --- Inicializace stavu ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "system_info" not in st.session_state:
    st.session_state.system_info = ""
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = DEFAULT_OLLAMA_HOST

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Nastavení")
    st.session_state.ollama_host = st.text_input(
        "🔗 Ollama API adresa",
        value=st.session_state.ollama_host,
        help="Adresa běžící Ollama instance (např. http://localhost:11434)"
    )
    available_models = load_available_models(st.session_state.ollama_host)
    if not available_models:
        available_models = ["Nebyly nalezeny modely"]
        st.warning("⚠️ Nelze načíst modely z Ollama. Zkontrolujte adresu a zda Ollama běží.")
        selected_model_index = 0
        st.session_state.selected_model = ""
    else:
        if st.session_state.selected_model in available_models:
            selected_model_index = available_models.index(st.session_state.selected_model)
        else:
            selected_model_index = 0
            st.session_state.selected_model = available_models[0]
    st.session_state.selected_model = st.selectbox(
        "🧠 Vyber model",
        options=available_models,
        index=selected_model_index,
        help="Modely dostupné na vaší Ollama instanci. Formát: název:digest"
    )
    temperature = st.slider("🌡️ Teplota (kreativita)", 0.0, 2.0, 0.7, 0.1,
                            help="Nižší hodnota = více deterministické, vyšší = kreativnější.")
    max_tokens = st.slider("🔢 Max. tokenů v odpovědi", 50, 4096, 512, 50,
                           help="Maximální délka generované odpovědi.")
    with st.expander("⚙️ Pokročilé nastavení"):
        cz_mode = st.checkbox("🇨🇿 Vynutit češtinu", value=False, help="Přidá instrukci modelu, aby odpovídal česky.")
        stream_mode = st.checkbox("🌀 Streamování odpovědí", value=True, help="Zobrazuje odpověď postupně, jak je generována.")
        show_tokens = st.checkbox("🔢 Zobrazovat tokeny", value=True, help="Zobrazí přibližný počet tokenů v odpovědi.")
    if st.button("❌ Vymazat historii chatu"):
        st.session_state.chat_history = []
        st.rerun()

with st.sidebar.expander("🧠 Aktuální paměť"):
    st.markdown(st.session_state.get("memory_summary", "_Paměť je zatím prázdná._"))


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
            summary = wikipedia.summary(search_term, sentences=3, auto_suggest=True)
            st.markdown("#### Shrnutí článku")
            st.markdown(f"""
                <div style="font-size: 0.85rem; line-height: 1.4; color: #cccccc; background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem;">
                    {summary}
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            include_wiki = st.checkbox("📎 Připojit shrnutí k dalšímu promptu?", key="wiki_include")
            if include_wiki:
                wiki_context = summary
        except wikipedia.exceptions.PageError:
            st.error(f"❌ Stránka '{search_term}' nebyla nalezena.")
        except wikipedia.exceptions.DisambiguationError as e:
            st.warning(f"🔀 '{search_term}' má více významů. Možnosti: {e.options[:5]}")
        except Exception as e:
            st.error(f"❌ Neočekávaná chyba při hledání na Wikipedii: {e}")
            logging.error(f"Wikipedia search error for '{search_term}': {e}")

# 📁 File Upload Plugin
with tab3:
    st.subheader("📂 Nahrát soubor pro kontext")
    uploaded_file = st.file_uploader("Vyber soubor (TXT, PDF, DOCX, CSV)", type=["txt", "pdf", "docx", "csv"], key="file_uploader")
    if uploaded_file is not None:
        try:
            file_type = uploaded_file.name.split(".")[-1].lower()
            bytes_data = uploaded_file.getvalue()
            if file_type == "txt":
                try:
                    file_context = bytes_data.decode("utf-8")
                except UnicodeDecodeError:
                    file_context = bytes_data.decode("latin-1")
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
                include_file = st.checkbox("📎 Připojit obsah souboru k dalšímu promptu?", key="file_include")
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
            page.raise_for_status()
            if 'text/html' in page.headers.get('Content-Type', ''):
                soup = BeautifulSoup(page.content, "html.parser")
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'article', 'section'])
                extracted_text = "\n".join(t.get_text(separator=' ', strip=True) for t in texts if t.get_text(strip=True))
                if extracted_text.strip():
                    include_web = st.checkbox("📎 Připojit text z webu k dalšímu promptu?", key="web_include")
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
    if "memory_summary" in st.session_state and st.session_state.memory_summary:
        context_history.append({"role": "system", "content": st.session_state.memory_summary})
    if cz_mode:
        context_history.append({"role": "system", "content": "Odpovídej výhradně v českém jazyce. Nemusíš zmiňovat, že odpovídáš česky."})
    for sender, message, _ in st.session_state.chat_history:
        role = "user" if sender == "user" else "assistant"
        core_message = message.split("\n\n*Přibližný počet tokenů:")[0].strip()
        context_history.append({"role": role, "content": core_message})
    
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
    selected_model_full = st.session_state.selected_model
    parts = selected_model_full.split(':')
    model_name_for_api = f"{parts[0]}:{parts[1]}" if len(parts) > 2 else selected_model_full
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
                "num_predict": max_tokens
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
            fallback_prompt = final_prompt
            if "memory_summary" in st.session_state and st.session_state.memory_summary:
                fallback_prompt = st.session_state.memory_summary + "\n\n" + fallback_prompt
            if cz_mode:
                fallback_prompt = "Odpovídej výhradně v českém jazyce. Nemusíš zmiňovat, že odpovídáš česky.\n\n" + fallback_prompt
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
            token_count = len(tokenizer.encode(final_answer)) if tokenizer else len(final_answer.split())
            if show_tokens:
                final_answer += f"\n\n*Přibližný počet tokenů: {token_count}*"
            st.session_state.chat_history.append(("bot", final_answer, datetime.now().strftime("%H:%M:%S")))
            
            # 🧠 Aktualizuj paměť po každé nové zprávě
            memory_summary = generate_combined_memory(st.session_state.chat_history)
            st.session_state.memory_summary = memory_summary
            logging.info("📥 Paměť aktualizována po nové zprávě.")

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