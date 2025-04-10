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
import time

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

#--- Konfigurace ---
ALLOWED_FILE_TYPES = ["txt", "pdf", "docx", "csv", "py", "html", "css", "json", "md"]
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_VERSION_URL = "/api/version"
OLLAMA_TAGS_URL = "/api/tags"
OLLAMA_GENERATE_URL = "/api/generate"
OLLAMA_CHAT_URL = "/api/chat" # Použijeme /api/chat pro messages
TOKENIZER_NAME = "gpt2"
# Zvýšený limit pro nedávné zprávy, protože /api/chat je efektivnější
RECENT_MESSAGES_TOKEN_LIMIT = 2000 # Např. 2000 tokenů
HISTORY_SUMMARIZATION_THRESHOLD = 10 # Po kolika zprávách sumarizovat
RECENT_MESSAGES_COUNT_LIMIT = 10 # Kolik posledních zpráv vzít (jako další limit)

try:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
    logging.info("Tokenizer byl úspěšně inicializován.")
except ImportError:
    logging.warning("Balíček transformers není nainstalován. Používá se fallback metoda.")
    tokenizer = None
except Exception as e:
    logging.error(f"Chyba při inicializaci tokenizeru: {e}")
    tokenizer = None

def count_tokens(text: str) -> int:
    if text is None:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logging.warning(f"Chyba tokenizace textu: {text[:50]}... Error: {e}")
            # Fallback na dělení podle mezer v případě chyby tokenizeru
            return len(text.split())
    # Fallback: jednoduché dělení textu na mezery
    return len(text.split())

# --- Pomocné funkce ---
def safe_get_digest(model_data: dict) -> str:
    try:
        digest = model_data.get('digest', '')
        if digest and ':' in digest:
            return digest.split(':')[-1][:7] # Zobrazíme 7 znaků hashe
        return digest[:7] if digest else "latest"
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
        attrs = match.group(1) # Zachytí atributy uvnitř tagu <think ...>
        content = match.group(2).strip() # Zachytí obsah mezi <think> a </think>

        # Hledá time="xx.x" v atributech
        time_match = re.search(r'time=["\']?([\d.]+)["\']?', attrs)
        time_text = f"{float(time_match.group(1)):.1f} sekundy" if time_match else "?"

        return f'''
        <details class="think-box">
            <summary>💭 Přemýšlení modelu (trvalo {time_text}) – klikni pro zobrazení</summary>
            <div class="think-content">{content}</div>
        </details>
        '''
    # Přidána podpora pro <thinking> tag
    text = re.sub(r'<thinking(.*?)>(.*?)</thinking>', replacer, text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r'<think(.*?)>(.*?)</think>', replacer, text, flags=re.DOTALL | re.IGNORECASE)

# --- Funkce pro paměť ---

# 🧠 Funkce pro shrnutí starší historie
def summarize_history(chat_history, model_name_full, host_url, max_summary_tokens=300, num_messages_to_summarize=10):
    """
    Summarizes a chunk of older chat history.
    Args:
        chat_history (list): The full chat history.
        model_name_full (str): The full name of the selected model (potentially with hash). # Přejmenováno pro jasnost
        host_url (str): The Ollama host URL.
        max_summary_tokens (int): Max tokens for the generated summary.
        num_messages_to_summarize (int): How many older messages to include in the summarization prompt.
    Returns:
        str or None: The generated summary or None if error/not needed.
    """
    if len(chat_history) <= HISTORY_SUMMARIZATION_THRESHOLD:
        logging.info("Historie je příliš krátká na sumarizaci.")
        return None # Není co sumarizovat

    # Vezmeme starší zprávy, ale ne úplně nejstarší, abychom sumarizovali relevantní blok
    # Např. vezmeme zprávy od -15 do -5 (10 zpráv)
    start_index = max(0, len(chat_history) - num_messages_to_summarize - RECENT_MESSAGES_COUNT_LIMIT)
    end_index = len(chat_history) - RECENT_MESSAGES_COUNT_LIMIT
    history_to_summarize = chat_history[start_index:end_index]

    if not history_to_summarize:
        logging.info("Nebyly nalezeny vhodné zprávy k sumarizaci.")
        return None

    logging.info(f"Sumarizace {len(history_to_summarize)} zpráv (indexy {start_index}-{end_index-1}).")

    to_summarize_text = ""
    for sender, message, timestamp in history_to_summarize:
        role = "Uživatel" if sender == "user" else "Asistent"
        # Odstraníme info o tokenech, pokud tam je
        content = re.sub(r'\n\n\*Přibližný počet tokenů: \d+\*$', '', message).strip()
        to_summarize_text += f"{timestamp} - {role}: {content}\n"

    # Odstraníme hash z názvu modelu pro API volání sumarizace
    parts = model_name_full.split(':')
    if len(parts) > 2:
        model_name_for_api = f"{parts[0]}:{parts[1]}"
        logging.info(f"Using model name for summarization API (stripped hash): {model_name_for_api}")
    else:
        model_name_for_api = model_name_full
        logging.info(f"Using model name for summarization API (no hash found): {model_name_for_api}")

    # Použijeme /api/generate pro sumarizaci (jednodušší prompt)
    prompt = f"Briefly summarize the key points from the following conversation in no more than {max_summary_tokens // 2} words. Focus on facts, decisions and important issues. Ignore greetings and casual conversation.:\n\n{to_summarize_text}"

    payload = {
        "model": model_name_for_api,
        "prompt": prompt,
        "options": {
            "temperature": 0.3, # Nižší teplota pro faktickou sumarizaci
            "num_predict": max_summary_tokens
        },
        "stream": False
    }
    api_url = f"{host_url.strip('/')}{OLLAMA_GENERATE_URL}"

    try:
        logging.info(f"Odesílání požadavku na sumarizaci na {api_url} s modelem {model_name_for_api}") # Logujeme správné jméno
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status() # Vyvolá chybu pro 4xx/5xx
        summary = response.json().get("response", "").strip()
        logging.info(f"Sumarizace úspěšná, délka: {len(summary)} znaků.")
        # Uložíme i čas vytvoření shrnutí
        return f"Shrnutí konverzace ({datetime.now().strftime('%H:%M')}):\n{summary}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Chyba při volání API pro sumarizaci ({api_url}): {e}")
    except Exception as e:
        logging.error(f"Neočekávaná chyba při sumarizaci: {e}")
    return None


# 🧠 Funkce pro získání posledních zpráv do tokenového limitu
def get_recent_messages_for_context(chat_history, token_limit=RECENT_MESSAGES_TOKEN_LIMIT, count_limit=RECENT_MESSAGES_COUNT_LIMIT):
    """
    Gets recent messages formatted for the /api/chat context, respecting token and count limits.
    Args:
        chat_history (list): Full chat history.
        token_limit (int): Maximum total tokens for recent messages.
        count_limit (int): Maximum number of recent messages to include.
    Returns:
        list: List of message dictionaries [{"role": ..., "content": ...}, ...]
    """
    messages = []
    total_tokens = 0
    # Vezmeme jen posledních 'count_limit' zpráv pro efektivitu
    recent_history = chat_history[-count_limit:]

    for sender, message, _ in reversed(recent_history):
        # Odstraníme info o tokenech a přemýšlení, pokud tam je
        content = re.sub(r'\n\n\*Přibližný počet tokenů: \d+\*$', '', message).strip()
        content = re.sub(r'<think.*?>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
        content = re.sub(r'<thinking.*?>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()

        if not content: # Přeskočíme prázdné zprávy po čištění
            continue

        token_count = count_tokens(content)

        if total_tokens + token_count > token_limit:
            logging.warning(f"Překročen tokenový limit ({token_limit}) pro krátkodobou paměť. Zahrnuto {len(messages)} zpráv.")
            break # Stop if adding this message exceeds the limit

        role = "user" if sender == "user" else "assistant"
        messages.insert(0, {"role": role, "content": content}) # Přidáváme na začátek, protože procházíme od konce
        total_tokens += token_count

    logging.info(f"Zahrnuto {len(messages)} nedávných zpráv ({total_tokens} tokenů) do kontextu.")
    return messages


# --- Inicializace stavu ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = st.session_state.get('model', None) # Start with None
if "system_info" not in st.session_state:
    st.session_state.system_info = ""
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = DEFAULT_OLLAMA_HOST
# Přidáme stav pro shrnutí paměti
if "memory_summary" not in st.session_state:
    st.session_state.memory_summary = None # Explicitně None na začátku

# --- Sidebar ---
with st.sidebar:
    st.subheader("⚙️ Nastavení")

    st.session_state.ollama_host = st.text_input(
        "🔗 Ollama API adresa",
        value=st.session_state.get("ollama_host", DEFAULT_OLLAMA_HOST),
        help="Adresa běžící Ollama instance (např. http://localhost:11434)"
    )

    # Tlačítko pro manuální načtení modelů
    if st.button("🔄 Načíst modely"):
        st.session_state["available_models"] = load_available_models(st.session_state.ollama_host)
        st.rerun() # Znovu načte stránku s novými modely

    # Načíst modely při startu stránky, pokud ještě nejsou načtené
    if "available_models" not in st.session_state:
        st.session_state["available_models"] = load_available_models(st.session_state.ollama_host)

    models = st.session_state.get("available_models", [])
    if not models:
        st.warning("⚠️ Nelze načíst modely z Ollama. Zkontrolujte adresu a zda Ollama běží.")
        models_display = ["Nelze načíst modely"]
        st.session_state.selected_model = None
    else:
        models_display = ["Zvolte model"] + models

    # Výběr modelu
    current_selection = st.session_state.get("selected_model")
    try:
        # Najdi index aktuálně vybraného modelu, pokud existuje a je platný
        if current_selection and current_selection in models_display:
             default_index = models_display.index(current_selection)
        else:
             default_index = 0 # Pokud není vybrán nebo není platný, vyber "Zvolte model"
    except ValueError:
        default_index = 0
        logging.warning(f"Model '{current_selection}' nenalezen v seznamu, resetuji výběr.")
        st.session_state.selected_model = None # Reset

    selected_model_value = st.selectbox(
        "🤖 Vyber model",
        models_display,
        index=default_index,
        key="selected_model_selector" # Použijeme jiný klíč pro selectbox
    )

    # Aktualizujeme hlavní stav modelu jen pokud se změnil a není to placeholder
    if selected_model_value != "Zvolte model" and selected_model_value != "Nelze načíst modely":
        st.session_state.selected_model = selected_model_value
    elif selected_model_value == "Zvolte model":
        # Pokud uživatel vybere "Zvolte model", resetujeme stav
        st.session_state.selected_model = None

    # Ostatní nastavení
    temperature = st.slider("🌡️ Teplota", 0.0, 1.0, st.session_state.get("temperature", 0.7), 0.05, key="temperature")
    max_tokens = st.slider("🔢 Max. tokenů v odpovědi", 128, 8192, st.session_state.get("max_tokens", 1024), 128, key="max_tokens", help="Maximální počet tokenů, které model vygeneruje v odpovědi.")
    # top_p = st.slider("📊 Top-p (nucleus sampling)", 0.0, 1.0, st.session_state.get("top_p", 0.9), 0.05, key="top_p") # Top-p méně používané s Ollama

    use_memory_checkbox = st.checkbox("🧠 Zapnout paměť (shrnutí + nedávné zprávy)", value=st.session_state.get("use_memory", True), key="use_memory")
    if use_memory_checkbox:
        st.caption("✅ Paměť aktivní: Bude použito shrnutí starší historie (pokud existuje) a nedávné zprávy.")
    else:
        st.caption("❌ Paměť vypnutá: Modelu bude poslán jen aktuální dotaz a případný kontext z pluginů.")

    cz_mode = st.checkbox("🇨🇿 Vynutit češtinu", value=st.session_state.get("cz_mode", False), key="cz_mode")
    stream_mode = st.checkbox("🌀 Streamování odpovědí", value=st.session_state.get("stream_mode", True), key="stream_mode")
    show_tokens_checkbox = st.checkbox("🔢 Zobrazovat tokeny v odpovědi", value=st.session_state.get("show_tokens", True), key="show_tokens")

    # Tlačítko pro vymazání historie a paměti
    if st.button("🗑️ Vymazat historii a paměť"):
        st.session_state.chat_history = []
        st.session_state.memory_summary = None
        logging.info("Chat history and memory summary cleared.")
        st.rerun()

    # Zobrazení systémových informací
    if st.session_state.get("system_info"):
        st.sidebar.caption(st.session_state.system_info)
    else:
        # Pokusíme se načíst info, pokud ještě není
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

# --- Hlavní část ---
st.title("🤖 Lokální LLM Chat")
# Zobrazíme aktuálně vybraný model nebo hlášku
display_model_name = st.session_state.selected_model if st.session_state.selected_model else "Není vybrán žádný model"
st.caption(f"Powered by Ollama & Streamlit | Model: {display_model_name}")

# Zobrazení historie chatu
st.markdown('<div id="chat-container" class="chat-container">', unsafe_allow_html=True)
if st.session_state.chat_history:
    for i, (sender, message, timestamp) in enumerate(st.session_state.chat_history):
        css_class = "user-message" if sender == "user" else "bot-message"
        icon = '🧑' if sender == 'user' else '🤖'
        sender_name = 'Ty' if sender == 'user' else 'LLM'
        # Použijeme unikátní klíč pro každý blok zprávy
        message_key = f"message_{i}_{sender}"
        # Formátování zprávy s časem a ikonou
        formatted_message = f"""
        <div class="{css_class}" key="{message_key}">
            <strong>{icon} {sender_name}</strong>
            <div style="margin-top: 0.3rem;">{format_thinking_block(message)}</div>
            <div style="font-size: 0.8rem; color: #888; text-align: right; margin-top: 0.5rem;">{timestamp}</div>
        </div>
        """
        st.markdown(formatted_message, unsafe_allow_html=True)
        # Přidáme malý oddělovač mezi zprávami pro lepší čitelnost
        st.markdown("<hr style='margin: 0.5rem 0; border-top: 1px solid #333;'>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Konec chat-containeru

# --- Vstupní oblast s pluginy ---
st.markdown('<div class="input-area">', unsafe_allow_html=True) # Začátek input area

# --- PLUGINY: Wikipedia, Soubory, Web ---
wiki_context = ""
file_context = ""
web_context = ""
include_wiki = False
include_file = False
include_web = False

# Použijeme sloupce pro přehlednější rozložení pluginů pod chatem
col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("📖 Wikipedia"):
        search_term = st.text_input("🔍 Heslo", key="wiki_search")
        if search_term:
            try:
                wikipedia.set_lang("cs")
                search_results = wikipedia.search(search_term, results=5) # Více výsledků
                if not search_results:
                    st.error(f"❌ Žádné výsledky pro '{search_term}'.")
                else:
                    # Dáme uživateli na výběr, který výsledek chce
                    selected_page = st.radio("Nalezené stránky:", search_results, key="wiki_select")
                    if selected_page:
                        try:
                            page = wikipedia.page(selected_page, auto_suggest=False) # Přesnější název
                            summary = wikipedia.summary(selected_page, sentences=5, auto_suggest=False) # Delší shrnutí
                            st.markdown("**Shrnutí:**")
                            st.markdown(f"<div class='plugin-content'>{summary}</div>", unsafe_allow_html=True)
                            if st.button(f"Použít shrnutí '{selected_page}'", key="wiki_use_button"):
                                include_wiki = True
                                wiki_context = summary
                                st.success(f"✅ Shrnutí '{selected_page}' bude přidáno.")

                        except wikipedia.exceptions.DisambiguationError as e:
                            st.warning(f"🔀 '{selected_page}' má více významů. Zkuste upřesnit hledání.")
                            st.caption(f"Možnosti: {e.options[:5]}")
                        except wikipedia.exceptions.PageError:
                            st.error(f"❌ Stránka '{selected_page}' nebyla nalezena.")

            except Exception as e:
                st.error(f"❌ Chyba při hledání na Wikipedii: {e}")
                logging.error(f"Wikipedia search error for '{search_term}': {e}")

with col2:
    with st.expander("📂 Soubor"):
        uploaded_file = st.file_uploader(
            f"Nahrát soubor ({', '.join(ALLOWED_FILE_TYPES)})",
            type=ALLOWED_FILE_TYPES,
            key="file_uploader"
        )
        if uploaded_file is not None:
            try:
                file_type = uploaded_file.name.split(".")[-1].lower()
                bytes_data = uploaded_file.getvalue()
                raw_text = "" # Inicializace

                # Zpracování podle typu souboru
                if file_type == "txt":
                    try:
                        raw_text = bytes_data.decode("utf-8")
                    except UnicodeDecodeError:
                        raw_text = bytes_data.decode("latin-1", errors='replace')
                elif file_type == "pdf":
                    with fitz.open(stream=bytes_data, filetype="pdf") as doc:
                        raw_text = "\n".join([page.get_text() for page in doc])
                elif file_type == "docx":
                    doc = docx.Document(io.BytesIO(bytes_data))
                    raw_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                elif file_type == "csv":
                    # Zkusíme různá kódování a oddělovače
                    try:
                        df = pd.read_csv(io.BytesIO(bytes_data))
                    except UnicodeDecodeError:
                        df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin-1')
                    except pd.errors.ParserError:
                        try:
                           df = pd.read_csv(io.BytesIO(bytes_data), sep=';')
                        except Exception:
                           df = pd.read_csv(io.BytesIO(bytes_data), sep=';', encoding='latin-1')
                    raw_text = df.to_string()
                elif file_type in ["py", "html", "css", "json", "md"]:
                     try:
                        raw_text = bytes_data.decode("utf-8")
                     except UnicodeDecodeError:
                        raw_text = bytes_data.decode("latin-1", errors='replace')
                     # Zvýraznění syntaxe pro náhled
                     try:
                         lexer = get_lexer_by_name(file_type, stripall=True)
                         formatter = HtmlFormatter(style="monokai", cssclass="syntaxhighlight", linenos=False) # Můžeme přidat čísla řádků: linenos='table'
                         highlighted_code = highlight(raw_text, lexer, formatter)
                         st.markdown("**Náhled kódu/struktury:**")
                         st.markdown(f"<div class='plugin-content code-block'>{highlighted_code}</div>", unsafe_allow_html=True)
                     except Exception as highlight_err:
                         st.warning(f"Chyba při zvýrazňování syntaxe: {highlight_err}")
                         st.markdown("**Náhled textu:**")
                         st.text_area("Obsah:", raw_text[:1000]+"...", height=150, disabled=True, key=f"file_preview_{uploaded_file.id}")
                else:
                    # Ostatní textové formáty zobrazíme jako prostý text
                    try:
                        raw_text = bytes_data.decode("utf-8")
                    except UnicodeDecodeError:
                        raw_text = bytes_data.decode("latin-1", errors='replace')
                    st.markdown("**Náhled textu:**")
                    st.text_area("Obsah:", raw_text[:1000]+"...", height=150, disabled=True, key=f"file_preview_{uploaded_file.id}")


                if raw_text.strip():
                    file_context = raw_text # Uložíme celý text pro kontext
                    if st.button(f"Použít obsah '{uploaded_file.name}'", key="file_use_button"):
                        include_file = True
                        st.success(f"✅ Obsah '{uploaded_file.name}' bude přidán.")
                else:
                    st.warning("⚠️ Obsah souboru je prázdný nebo jej nelze extrahovat.")

            except Exception as e:
                st.error(f"❌ Chyba při zpracování souboru '{uploaded_file.name}': {e}")
                logging.error(f"Error processing file '{uploaded_file.name}': {e}\n{traceback.format_exc()}")

with col3:
    with st.expander("🌍 Web"):
        web_url = st.text_input("🔗 URL adresa", key="web_url")
        if web_url:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                page = requests.get(web_url, timeout=15, headers=headers, allow_redirects=True)
                page.raise_for_status() # Zkontroluje HTTP chyby

                content_type = page.headers.get('Content-Type', '').lower()

                if 'html' in content_type:
                    soup = BeautifulSoup(page.content, "html.parser")
                    # Odstranění nepotřebných tagů
                    for element in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "iframe", "noscript"]):
                        element.decompose()

                    # Pokus o extrakci hlavního obsahu (časté tagy/třídy)
                    main_content = soup.find('article') or soup.find('main') or soup.find(role='main') or soup.find(class_=re.compile("content|post|entry|article")) or soup.body

                    if main_content:
                       texts = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'pre', 'code'], limit=200)
                       extracted_text = "\n".join(t.get_text(separator=' ', strip=True) for t in texts if t.get_text(strip=True))
                    else: # Fallback na celý body, pokud se nepodařilo najít hlavní obsah
                       texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'], limit=200)
                       extracted_text = "\n".join(t.get_text(separator=' ', strip=True) for t in texts if t.get_text(strip=True))


                    if extracted_text.strip():
                        st.markdown("**Extrahovaný text (max 1000 znaků):**")
                        st.markdown(f"<div class='plugin-content'>{extracted_text[:1000]}...</div>", unsafe_allow_html=True)
                        if st.button("Použít text z webu", key="web_use_button"):
                             include_web = True
                             web_context = extracted_text # Celý extrahovaný text
                             st.success("✅ Text z webu bude přidán.")
                    else:
                        st.warning("⚠️ Nepodařilo se extrahovat text z hlavních tagů stránky.")
                else:
                    st.warning(f"⚠️ Obsah na URL není HTML (typ: {content_type}). Zkuste jinou stránku.")

            except requests.exceptions.Timeout:
                 st.error("❌ Časový limit vypršel při načítání URL.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Chyba při načítání URL: {e}")
                logging.error(f"Error fetching URL '{web_url}': {e}")
            except Exception as e:
                st.error(f"❌ Neočekávaná chyba při zpracování webu: {e}")
                logging.error(f"Error processing web content from '{web_url}': {e}\n{traceback.format_exc()}")

# --- Formulář pro odeslání chatu ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("💬 Váš dotaz:", height=100, key="user_input", placeholder="Zadejte dotaz nebo instrukci...")
    submitted = st.form_submit_button("🚀 Odeslat")

st.markdown('</div>', unsafe_allow_html=True) # Konec input area

# --- Zpracování odeslaného formuláře ---
if submitted and user_input.strip():
    if not st.session_state.selected_model:
        st.warning("⚠️ Prosím, vyberte model v nastavení v postranním panelu.")
    else:
        # 1. Přidání vstupu uživatele do historie
        timestamp = datetime.now().strftime("%H:%M:%S")
        user_message_content = user_input.strip()
        st.session_state.chat_history.append(("user", user_message_content, timestamp))
        logging.info(f"Uživatel ({timestamp}): {user_message_content}")

        # 2. Příprava kontextu pro model
        messages_for_api = []
        final_prompt_for_fallback = [] # Pro fallback /api/generate

        # a) Systémová zpráva (vždy první, pokud je potřeba)
        system_prompt = ""
        if st.session_state.cz_mode:
            system_prompt = "Jsi AI asistent. Odpovídej výhradně v českém jazyce. Buď stručný a věcný."
        # Přidáme systémovou zprávu, pokud není prázdná
        if system_prompt:
            messages_for_api.append({"role": "system", "content": system_prompt})
            final_prompt_for_fallback.append(f"SYSTEM: {system_prompt}")

        # b) Kontext z pluginů (přidáme před paměť)
        plugin_context_text = ""
        if include_wiki and wiki_context:
            plugin_context_text += f"The context from Wikipedia::\n{wiki_context}\n\n"
        if include_file and file_context:
            # Omezíme délku pro jistotu
            file_context_limited = file_context[:15000]
            plugin_context_text += f"The context from the file ({uploaded_file.name}):\n{file_context_limited}\n{'...' if len(file_context) > 15000 else ''}\n\n"
        if include_web and web_context:
            # Omezíme délku
            web_context_limited = web_context[:15000]
            plugin_context_text += f"The context from website ({web_url}):\n{web_context_limited}\n{'...' if len(web_context) > 15000 else ''}\n\n"

        # Přidáme kontext z pluginů jako jednu systémovou zprávu (nebo uživatelskou?) - zkusíme user
        if plugin_context_text:
            # messages_for_api.append({"role": "system", "content": f"Doplňující kontext:\n{plugin_context_text.strip()}"})
            # Nebo raději jako první "user" zprávu, aby model věděl, že je to vstup pro aktuální dotaz
            messages_for_api.append({"role": "user", "content": f"Use the following context to answer:\n{plugin_context_text.strip()}"})
            final_prompt_for_fallback.append(f"CONTEXT:\n{plugin_context_text.strip()}")


        # c) Paměť (pokud je zapnutá)
        if st.session_state.use_memory:
            # i) Dlouhodobá paměť (shrnutí)
            # Zkusíme sumarizovat, pokud je historie dost dlouhá a shrnutí ještě neexistuje nebo je staré
            if len(st.session_state.chat_history) > HISTORY_SUMMARIZATION_THRESHOLD + 1 and not st.session_state.memory_summary:
                 # TODO: Možná přidat časovou podmínku pro re-sumarizaci?
                 logging.info("Pokus o sumarizaci historie...")
                 model_name_for_api = st.session_state.selected_model # Celé jméno modelu
                 summary = summarize_history(
                     st.session_state.chat_history[:-1], # Sumarizujeme historii *před* posledním dotazem uživatele
                     st.session_state.selected_model,
                     st.session_state.ollama_host
                 )
                 if summary:
                     st.session_state.memory_summary = summary
                 else:
                     logging.warning("Sumarizace se nezdařila nebo nebyla potřeba.")


            # Přidáme existující shrnutí do kontextu
            if st.session_state.memory_summary:
                messages_for_api.append({"role": "system", "content": st.session_state.memory_summary})
                final_prompt_for_fallback.append(f"SUMMARY:\n{st.session_state.memory_summary}")

            # ii) Krátkodobá paměť (nedávné zprávy)
            recent_messages = get_recent_messages_for_context(
                st.session_state.chat_history[:-1] # Poslední zprávy *před* aktuálním dotazem uživatele
            )
            messages_for_api.extend(recent_messages) # Přidáme je do seznamu zpráv
            # Přidáme nedávné zprávy i do fallback promptu
            if recent_messages:
                 recent_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_messages])
                 final_prompt_for_fallback.append(f"RECENT HISTORY:\n{recent_text}")

        # d) Aktuální dotaz uživatele (jako poslední uživatelská zpráva)
        messages_for_api.append({"role": "user", "content": user_message_content})
        final_prompt_for_fallback.append(f"USER QUERY:\n{user_message_content}")

        # 3. Sestavení payloadu a volání API
        selected_model_full = st.session_state.selected_model
        # Odstraníme hash (část za druhou dvojtečkou), pokud existuje
        parts = selected_model_full.split(':')
        if len(parts) > 2:
            model_name_for_api = f"{parts[0]}:{parts[1]}"
            logging.info(f"Using model name for API (stripped hash): {model_name_for_api}")
        else:
            # Pokud tam hash není (např. "llama3:latest"), použijeme celé jméno
            model_name_for_api = selected_model_full
            logging.info(f"Using model name for API (no hash found): {model_name_for_api}")

        api_url = f"{st.session_state.ollama_host.strip('/')}{OLLAMA_CHAT_URL}"
        payload = {
            "model": model_name_for_api,
            "messages": messages_for_api,
            "options": {
                "temperature": st.session_state.temperature,
                "num_predict": st.session_state.max_tokens,
                # "top_p": st.session_state.top_p, # Můžeme přidat, pokud chceme
            },
            "stream": st.session_state.stream_mode
        }

        # --- DEBUG: Zobrazení odesílaného kontextu ---
        with st.expander("🕵️ Kontext odeslaný modelu (klikni pro zobrazení)", expanded=False):
            st.json(payload) # Zobrazíme celý JSON payload
            # Můžeme přidat i odhad celkového počtu tokenů kontextu
            context_tokens = sum(count_tokens(msg.get("content", "")) for msg in messages_for_api)
            st.caption(f"Odhadovaný počet tokenů v odeslaném kontextu: {context_tokens}")
        # --- Konec DEBUG sekce ---

        final_answer = ""
        response_successful = False
        response_duration = 0 # Inicializace doby odezvy

        try:
            logging.info(f"Odesílání požadavku na {api_url} s modelem {model_name_for_api}")
            start_time = time.time() # <-- ZAČÁTEK MĚŘENÍ ČASU
            with st.spinner("🧠 Přemýšlím..."):
                response = requests.post(api_url, json=payload, stream=st.session_state.stream_mode, timeout=120) # Timeout 120s
                response.raise_for_status() # Chyby 4xx/5xx vyvolají výjimku

                if st.session_state.stream_mode:
                    # Streamování odpovědi
                    placeholder = st.empty()
                    assistant_response_content = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if "message" in chunk and "content" in chunk["message"]:
                                     token = chunk["message"]["content"]
                                     assistant_response_content += token
                                     # Aktualizujeme obsah placeholderu s formátováním Markdown
                                     placeholder.markdown(f"**🤖 LLM:**\n{assistant_response_content}")

                                # Zkontrolujeme, zda je streamování dokončeno
                                if chunk.get("done"):
                                    end_time = time.time() # <-- KONEC MĚŘENÍ (stream)
                                    response_duration = end_time - start_time
                                    logging.info(f"Streamování dokončeno. Trvání: {response_duration:.2f}s")
                                    final_answer = assistant_response_content
                                    response_successful = True
                                    # Zkontrolujeme HTTP status až po dokončení streamu, pokud je to možné
                                    # V Ollama streamu není status code v každém chunku.
                                    # Pokud by response objekt sám měl status code, zkontrolovali bychom ho zde.
                                    # Prozatím spoléháme na raise_for_status() níže, i když pro stream nemusí být ideální.
                                    # response.raise_for_status() # Může způsobit problémy se streamem, pokud chyba nastane později
                                    if response.status_code != 200:
                                         logging.error(f"Stream finished but with error status: {response.status_code} {response.reason}")
                                         # Můžeme zde vyvolat výjimku nebo nastavit response_successful = False
                                         response.raise_for_status() # Zkusíme vyvolat chybu                                    
                                    break # Ukončíme smyčku

                            except json.JSONDecodeError as json_err:
                                logging.error(f"Chyba dekódování JSON streamu: {json_err} - Data: {line}")
                                final_answer += f"\n[Chyba v datech streamu: {line}]"
                            except Exception as stream_err:
                                logging.error(f"Neočekávaná chyba při zpracování streamu: {stream_err}")
                                final_answer += f"\n[Chyba streamu: {stream_err}]"
                else:
                    # Bez streamování
                    response.raise_for_status() # Zkontrolujeme chybu hned
                    end_time = time.time() # <-- KONEC MĚŘENÍ (non-stream)
                    response_duration = end_time - start_time
                    logging.info(f"Odpověď přijata (non-stream). Trvání: {response_duration:.2f}s")
                    data = response.json()
                    if "message" in data and "content" in data["message"]:
                        final_answer = data["message"]["content"]
                        response_successful = True
                    else:
                        logging.error(f"Odpověď API neobsahuje očekávanou strukturu ('message': 'content'): {data}")
                        st.error("Chyba: Odpověď serveru neobsahovala text odpovědi.")

        except requests.exceptions.Timeout:
            end_time = time.time()
            response_duration = end_time - start_time
            st.error("❌ Časový limit vypršel při čekání na odpověď od Ollama.")
            logging.error(f"Ollama API request timed out after {response_duration:.2f}s.")
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            response_duration = end_time - start_time
            st.error(f"❌ Chyba komunikace s Ollama API ({api_url}): {e}")
            logging.error(f"Ollama API request error after {response_duration:.2f}s: {e}\n{traceback.format_exc()}")
        except Exception as e:
            end_time = time.time()
            response_duration = end_time - start_time
            st.error(f"⚠️ Neočekávaná chyba při generování odpovědi: {e}")
            logging.error(f"Unexpected error during Ollama call after {response_duration:.2f}s: {e}\n{traceback.format_exc()}")


        # --- FALLBACK na /api/generate POKUD /api/chat selhal ---
        if not response_successful:
            logging.warning(f"Primární volání /api/chat selhalo nebo nevrátilo odpověď (trvalo {response_duration:.2f}s), zkouším fallback na /api/generate.")
            api_url_fallback = f"{st.session_state.ollama_host.strip('/')}{OLLAMA_GENERATE_URL}"
            logging.info(f"Fallback URL sestaveno jako: {api_url_fallback}") # Přidejte tento log
            fallback_prompt_text = "\n\n".join(final_prompt_for_fallback) # Spojíme části promptu

            payload_fallback = {
                "model": model_name_for_api,
                "prompt": fallback_prompt_text,
                "options": {
                    "temperature": st.session_state.temperature,
                    "num_predict": st.session_state.max_tokens
                },
                "stream": st.session_state.stream_mode # Použijeme stejný režim streamování
            }

            # --- DEBUG: Zobrazení fallback kontextu ---
            with st.expander("🕵️ Fallback Kontext odeslaný modelu (/api/generate)", expanded=False):
                 st.json(payload_fallback)
                 context_tokens_fallback = count_tokens(fallback_prompt_text)
                 st.caption(f"Odhadovaný počet tokenů ve fallback kontextu: {context_tokens_fallback}")
            # --- Konec DEBUG sekce ---

            final_answer = "" # Resetujeme final_answer pro fallback
            response_duration = 0 # Resetujeme i duration pro fallback měření

            try:
                logging.info(f"Odesílání fallback požadavku na {api_url_fallback} s modelem {model_name_for_api}")
                start_time = time.time() # <-- ZAČÁTEK MĚŘENÍ (fallback)
                with st.spinner("🧠 Přepínám na nouzový režim (/api/generate)..."):
                    response_fallback = requests.post(api_url_fallback, json=payload_fallback, stream=st.session_state.stream_mode, timeout=120)
                    response_fallback.raise_for_status()

                    if st.session_state.stream_mode:
                        placeholder = st.empty()
                        assistant_response_content = ""
                        for line in response_fallback.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line.decode())
                                    token = data.get("response", "")
                                    assistant_response_content += token
                                    placeholder.markdown(f"**🤖 LLM (Fallback):**\n{assistant_response_content}")
                                    if data.get("done"):
                                        end_time = time.time() # <-- KONEC MĚŘENÍ (fallback stream)
                                        response_duration = end_time - start_time
                                        logging.info(f"Fallback stream dokončen. Trvání: {response_duration:.2f}s")
                                        final_answer = assistant_response_content
                                        # Zde bychom také měli zkontrolovat status, pokud to jde
                                        if response_fallback.status_code != 200:
                                            response_fallback.raise_for_status()
                                        response_successful = True # I fallback se počítá jako úspěch
                                        break
                                except Exception as e:
                                    logging.error(f"Chyba zpracování fallback streamu: {e} - Data: {line}")
                                    final_answer += f"\n[Chyba fallback streamu: {line}]"
                    else:
                        response_fallback.raise_for_status()
                        end_time = time.time() # <-- KONEC MĚŘENÍ (fallback non-stream)
                        response_duration = end_time - start_time
                        logging.info(f"Fallback odpověď přijata (non-stream). Trvání: {response_duration:.2f}s")
                        data = response_fallback.json()
                        final_answer = data.get("response", "")
                        if final_answer:
                            response_successful = True
                        else:
                            logging.error(f"Fallback odpověď API neobsahuje 'response': {data}")
                            st.error("Chyba: Fallback odpověď serveru neobsahovala text.")

            except requests.exceptions.Timeout:
                end_time = time.time()
                response_duration = end_time - start_time
                st.error("❌ Časový limit vypršel i při fallback požadavku.")
                logging.error(f"Ollama API fallback request timed out after {response_duration:.2f}s.")
            except requests.exceptions.RequestException as e:
                end_time = time.time()
                response_duration = end_time - start_time
                st.error(f"❌ Chyba komunikace s Ollama API i při fallbacku ({api_url_fallback}): {e}")
                logging.error(f"Ollama API fallback request error after {response_duration:.2f}s: {e}\n{traceback.format_exc()}")
            except Exception as e:
                end_time = time.time()
                response_duration = end_time - start_time
                st.error(f"⚠️ Neočekávaná chyba při fallback generování: {e}")
                logging.error(f"Unexpected error during Ollama fallback call after {response_duration:.2f}s: {e}\n{traceback.format_exc()}")


        # 4. Zpracování finální odpovědi (pokud nějaká je)
        if final_answer.strip():
            bot_message_content_raw = final_answer.strip()
            bot_message_content_formatted = bot_message_content_raw # Výchozí formátovaná verze

            # --- START: Vložení času do think tagu, pokud existuje ---
            think_match = re.search(r"<(think|thinking)", bot_message_content_raw, re.IGNORECASE)
            if think_match and response_duration > 0:
                tag_name = think_match.group(1) # 'think' or 'thinking'
                # Použijeme re.sub s count=1 pro nahrazení pouze prvního výskytu
                # Nahradíme '<tag ...>' za '<tag time="X.Y" ...>'
                # Regex: r"<({tag_name})([^>]*)>" - najde otevírací tag a zachytí atributy
                # Nahrazení: r'<\1 time="{response_duration:.1f}"\2>' - vloží time atribut
                try:
                    bot_message_content_formatted = re.sub(
                        rf"<({tag_name})([^>]*)>",
                        rf'<\1 time="{response_duration:.1f}"\2>',
                        bot_message_content_raw,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    logging.info(f"Inserted time={response_duration:.1f}s into <{tag_name}> tag.")
                except Exception as regex_err:
                    logging.error(f"Error inserting time into think tag: {regex_err}")
                    # Pokud náhrada selže, použijeme původní text
                    bot_message_content_formatted = bot_message_content_raw
            # --- END: Vložení času do think tagu ---


            # Přidáme počet tokenů, pokud je zapnuto
            if st.session_state.show_tokens:
                token_count = count_tokens(bot_message_content_raw) # Počítáme tokeny z původní odpovědi
                # Přidáme info o tokenech k potenciálně upravenému obsahu (s vloženým časem)
                # Je důležité NEPŘIDÁVAT to PŘED voláním format_thinking_block
                # Takže to přidáme až NAKONEC
                # Tuto část přesuneme až za format_thinking_block

            timestamp = datetime.now().strftime("%H:%M:%S")

            # Zformátujeme POUZE pokud obsahuje think tag (jinak necháme plain text)
            # Použijeme finální obsah s potenciálně vloženým časem
            final_display_content = format_thinking_block(bot_message_content_formatted)

            # Přidání informace o tokenech až teď, pokud je třeba
            if st.session_state.show_tokens:
                final_display_content += f"\n\n*Přibližný počet tokenů: {token_count}*"


            st.session_state.chat_history.append(("bot", final_display_content, timestamp))
            logging.info(f"LLM ({timestamp}): {bot_message_content_raw[:100]}...") # Logujeme jen začátek *původní* odpovědi
            # Vyčistíme použité pluginy pro další kolo (nepřenáší se automaticky)
            # Toto můžeme udělat i jinak, pokud chceme pluginy držet déle
            include_wiki = False
            include_file = False
            include_web = False
            st.rerun() # Znovu načte stránku a zobrazí novou zprávu
        elif response_successful:
             st.warning("⚠️ Model vrátil prázdnou odpověď.")
             logging.warning("Model returned an empty response despite successful API call.")
        else:
             # Pokud ani fallback neuspěl nebo nevrátil odpověď
             st.error("❌ Nepodařilo se získat odpověď od modelu ani nouzovým režimem.")
             logging.error("Failed to get response from model even with fallback.")


elif submitted and not user_input.strip():
    st.warning("⚠️ Prosím, zadejte dotaz.")