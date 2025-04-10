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

st.set_page_config(
    page_title="LLM Chat – Ollama",
    page_icon="🤖",
    layout="wide",  # ✅ vynutí wide mode
    menu_items={'About': "### Lokální LLM chat pomocí Ollama a Streamlit"}
)

# Konfigurace
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_VERSION_URL = "/api/version"
OLLAMA_TAGS_URL = "/api/tags"
OLLAMA_GENERATE_URL = "/api/generate"
TOKENIZER_NAME = "gpt2"

# Inicializace tokenizeru
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
except Exception as e:
    st.error(f"Chyba při načítání tokenizeru: {e}")

# Funkce

def safe_get_digest(model_data):
    try:
        return model_data.get('digest', '').split(':')[-1][:6]
    except:
        return "unknown"

def load_available_models(host):
    try:
        response = requests.get(f"{host}{OLLAMA_TAGS_URL}", timeout=10)
        if response.status_code == 200:
            models_data = response.json().get('models', [])
            return [
                f"{model['name']}:{safe_get_digest(model)}" 
                for model in models_data
            ]
        return []
    except Exception as e:
        st.error(f"Chyba při načítání modelů: {e}")
        return []

# Styl
st.markdown("""
<style>
    .user-message {
        background-color: #2e2e2e;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        border: 1px solid #444;
        margin-top: 0.5rem;
        width: fit-content;        /* Přidáno – element se roztáhne podle obsahu */
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.5;
        margin-left: auto;         /* zarovnání doprava */
        margin-right: 0;
        display: block;
        text-align: right;
    }

    .bot-message {
        background-color: #1e1e1e;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        border: 1px solid #444;
        margin-top: 0.5rem;
        width: fit-content;        /* Přidáno – element se roztáhne podle obsahu */
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.5;
        margin-left: 0;
        margin-right: auto;
        display: block;
        text-align: left;
    }


    .user-message strong {
        color: #fdd835; /* zlatá */
    }

    .bot-message strong {
        color: #03dac6; /* tyrkysová */
    }

    .stMarkdown {
        padding: 0;
        margin: 0;
    }

    [data-testid="stSidebar"] {
        background-color: #111111 !important;
    }

    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }

    /* Scrollovatelné okno pro historii */
    .chat-container {
    max-height: 65vh;
    overflow-y: auto;
    margin-bottom: 1rem;
    }

</style>
""", unsafe_allow_html=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "system_info" not in st.session_state:
    st.session_state.system_info = ""
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = DEFAULT_OLLAMA_HOST

# Sidebar
with st.sidebar:
    
    st.title("⚙️ Nastavení")

    st.session_state.ollama_host = st.text_input(
        "🔗 Ollama API adresa", st.session_state.ollama_host
    )

    available_models = load_available_models(st.session_state.ollama_host)
    if not available_models:
        available_models = ["llama2-uncensored:latest"]
        st.warning("⚠️ Nebyly nalezeny žádné modely, použit fallback")

    try:
        default_index = available_models.index(st.session_state.selected_model) if st.session_state.selected_model else 0
    except (ValueError, IndexError):
        default_index = 0

    try:
        st.session_state.selected_model = st.selectbox(
            "🧠 Vyber model",
            options=available_models,
            index=default_index,
            help="Modely se načítají z Ollama instance"
        )
    except IndexError:
        st.session_state.selected_model = available_models[0]
        st.rerun()

    model_name = st.session_state.selected_model.lower()
    if ":32b" in model_name:
        st.warning("⚠️ Tento model vyžaduje hodně RAM (doporučeno 32GB+)")

    temperature = st.slider("🌡️ Teplota (kreativita)", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("🔢 Max. tokenů", 50, 4096, 512, 50)

    with st.expander("⚙️ Pokročilé nastavení"):
        cz_mode = st.checkbox("🇨🇿 Vynutit češtinu", value=False)
        stream_mode = st.checkbox("🌀 Streamování odpovědí", value=True)
        show_tokens = st.checkbox("🔢 Zobrazovat tokeny", value=True)

    st.markdown("---")
    if st.button("🔄 Aktualizovat modely"):
        available_models = load_available_models(st.session_state.ollama_host)
    if st.button("❌ Vymazat chat"):
        st.session_state.chat_history = []

# Titulek
st.title("🤖 Lokální LLM Chat")
st.caption("Powered by Ollama & Streamlit")

# PLUGINY: Wikipedia a soubory
wiki_context = ""
file_context = ""

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📖 Wikipedia", "📂 Soubor", "🌍 Web"])

# 🧭 Wikipedia
with tab2:
    st.subheader("📖 Wikipedia vyhledávání")
    search_term = st.text_input("🔍 Heslo pro vyhledání", placeholder="např. Umělá inteligence")

    if search_term:
        try:
            wikipedia.set_lang("cs")
            summary = wikipedia.summary(search_term, sentences=5)
            st.text_area("📝 Shrnutí článku", summary, height=200)
            include_wiki = st.checkbox("📎 Připojit k promptu?")
            if include_wiki:
                wiki_context = summary
        except wikipedia.exceptions.DisambiguationError as e:
            st.warning(f"🔀 Příliš obecné heslo, více významů: {e.options[:5]}")
        except Exception as e:
            st.error(f"❌ Nelze načíst článek: {e}")

# 📁 Soubor
with tab3:
    st.subheader("📂 Nahrát soubor")
    uploaded_file = st.file_uploader("Vyber soubor (TXT, PDF, DOCX, CSV)", type=["txt", "pdf", "docx", "csv"])

    include_file = False
    file_preview = ""

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()

        try:
            if file_type == "txt":
                file_preview = uploaded_file.read().decode("utf-8")

            elif file_type == "pdf":
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                file_preview = "\n".join([page.get_text() for page in doc])

            elif file_type == "docx":
                doc = docx.Document(uploaded_file)
                file_preview = "\n".join([para.text for para in doc.paragraphs])

            elif file_type == "csv":
                df = pd.read_csv(uploaded_file)
                file_preview = df.to_string()

            if file_preview.strip():
                st.text_area("📝 Náhled obsahu", file_preview[:3000], height=200)
                include_file = st.checkbox("📎 Připojit k promptu?")
                if include_file:
                    file_context = file_preview
            else:
                st.warning("⚠️ Soubor je prázdný nebo nelze načíst.")
        except Exception as e:
            st.error(f"❌ Chyba při čtení souboru: {e}")

# 🌐 Plugin: URL
with tab4:
    st.subheader("🌍 Načíst obsah z webu")
    web_url = st.text_input("🔗 Zadej URL adresu", placeholder="https://...")

    web_context = ""
    
    if web_url:
        try:
            page = requests.get(web_url, timeout=10)
            soup = BeautifulSoup(page.content, "html.parser")
            
            # Extrahujeme hlavně odstavce a nadpisy
            texts = soup.find_all(["h1", "h2", "h3", "p"])
            extracted_text = "\n".join(t.get_text(strip=True) for t in texts if t.get_text(strip=True))
            
            if extracted_text.strip():
                st.text_area("📝 Náhled obsahu", extracted_text[:3000], height=200)
                include_web = st.checkbox("📎 Připojit k promptu?")
                if include_web:
                    web_context = extracted_text
            else:
                st.warning("⚠️ Nepodařilo se extrahovat žádný čitelný text.")
        
        except Exception as e:
            st.error(f"❌ Chyba při načítání webu: {e}")

# 📝 Vstup od uživatele
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("💬 Napiš dotaz:", height=120, placeholder="Zadej prompt...")
    submitted = st.form_submit_button("🚀 Odeslat")

if submitted and user_input.strip():
    # Poskládání historie chatu pro kontext
    history_text = ""
    for sender, message, _ in st.session_state.chat_history:
        role = "Ty" if sender == "user" else "LLM"
        history_text += f"{role}: {message.strip()}\n"

    # Příprava aktuálního promptu
    prompt = user_input.strip()
    if cz_mode:
        prompt = "Odpovídej výhradně česky. " + prompt

    # Sestavení finálního promptu s historií
    prompt_with_history = f"{history_text}Ty: {prompt}"


    # Připojení kontextu z pluginů
    if wiki_context:
        prompt = f"[WIKIPEDIA KONTEXT]: {wiki_context}\n\n{prompt}"

    if file_context:
        prompt = f"[SOUBOR KONTEXT]: {file_context}\n\n{prompt}"

    if web_context:
        prompt = f"[WEB KONTEXT]: {web_context}\n\n{prompt}"

    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append(("user", prompt, timestamp))

    model_parts = st.session_state.selected_model.split(":")
    model_name = ":".join(model_parts[:-1]) if len(model_parts) > 1 else st.session_state.selected_model

    payload = {
        "model": model_name,
        "prompt": prompt_with_history,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": stream_mode
    }

    try:
        with st.spinner("🧠 Generuji odpověď..."):
            response = requests.post(
                f"{st.session_state.ollama_host}{OLLAMA_GENERATE_URL}",
                json=payload,
                stream=stream_mode,
                timeout=120
            )
            final_answer = ""

            if response.status_code == 200:
                if stream_mode:
                    placeholder = st.empty()
                    for line in response.iter_lines():
                        if line:
                            try:
                                decoded = line.decode()
                                if decoded.strip().startswith("{"):
                                    data = json.loads(decoded)
                                    token = data.get("response", "")
                                    final_answer += token
                                    placeholder.markdown(f"**🤖 LLM:** {final_answer}")
                            except json.JSONDecodeError:
                                st.warning("⚠️ Jeden řádek odpovědi nešel přečíst (JSON chyba)")
                else:
                    data = response.json()
                    final_answer = data.get("response", "")

                token_count = len(tokenizer.encode(final_answer)) if tokenizer else len(final_answer.split())
                if show_tokens:
                    final_answer += f"\n\n*Přibližný počet tokenů: {token_count}*"

                # ✅ ULOŽIT pouze správnou odpověď do historie
                st.session_state.chat_history.append(("bot", final_answer, datetime.now().strftime("%H:%M:%S")))
                st.rerun()

            else:
                st.error(f"❌ API odpovědělo s chybou: {response.status_code}")
    except Exception as e:
        st.error(f"⚠️ Chyba během komunikace s Ollama API: {e}")
        print(traceback.format_exc())  # pro ladění v terminálu

# Historie chatu
if st.session_state.chat_history:    
    st.markdown('<div id="chat-container" class="chat-container">', unsafe_allow_html=True)

    for sender, message, timestamp in reversed(st.session_state.chat_history):
        css_class = "user-message" if sender == "user" else "bot-message"

        formatted_message = f"""
<div class="{css_class}">
    <strong>{'🧑 Ty' if sender == 'user' else '🤖 LLM'}</strong><br>
    {message}
    <div style="font-size: 0.8rem; color: #888; margin-top: 0.5rem;">{timestamp}</div>
</div>
"""
        st.markdown(formatted_message, unsafe_allow_html=True)
        st.write("")

    st.markdown('</div>', unsafe_allow_html=True)
 


if not st.session_state.system_info:
    try:
        system_response = requests.get(f"{st.session_state.ollama_host}{OLLAMA_VERSION_URL}", timeout=5)
        if system_response.status_code == 200:
            version_info = system_response.json()
            st.session_state.system_info = f"Ollama verze: {version_info.get('version', 'N/A')}"
        else:
            st.warning("⚠️ Ollama odpověděla, ale status nebyl 200 OK.")
    except Exception as e:
        st.warning("⚠️ Nepodařilo se získat informace o verzi Ollamy.")

        
print(traceback.format_exc())
# Konec