# --- START OF FILE chat_llm.py ---

import streamlit as st
import requests
import json
from datetime import datetime
from transformers import AutoTokenizer

# 💾 Funkce pro vytvoření shrnutí z historie chatu (kombinovaný styl)
def generate_combined_memory(chat_history, tokenizer=None):
    if not chat_history:
        return ""
    diary_part = "🧠 Kontext z minulých událostí:\n\n"
    bullet_part = "\n\n**• Klíčové body:**\n"

    for sender, message, timestamp in chat_history:
        role = "Uživatel" if sender == "user" else "Asistent"
        clean = message.split("\n\n*Přibližný počet tokenů:")[0].strip()
        diary_part += f"{timestamp} – {role} řekl: {clean}\n"

    for sender, message, _ in chat_history[-5:]:  # posledních 5 pro body
        role = "Uživatel" if sender == "user" else "Asistent"
        clean = message.split("\n\n*Přibližný počet tokenů:")[0].strip()
        bullet_part += f"- {role}: {clean[:100]}{'...' if len(clean)>100 else ''}\n"

    return diary_part + bullet_part
import wikipedia
import pandas as pd
import docx
import fitz  # z PyMuPDF
import io
from bs4 import BeautifulSoup
import traceback
import logging # Added for better logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="LLM Chat – Ollama",
    page_icon="🤖",
    layout="wide",  # ✅ vynutí wide mode
    menu_items={'About': "### Lokální LLM chat pomocí Ollama a Streamlit"}
)

# --- Konfigurace ---
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_VERSION_URL = "/api/version"
OLLAMA_TAGS_URL = "/api/tags"
OLLAMA_GENERATE_URL = "/api/generate"
TOKENIZER_NAME = "gpt2" # Using a common tokenizer for approximation

# --- Inicializace tokenizeru ---
# Load tokenizer once globally, handle potential errors during startup
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    logging.info(f"Tokenizer '{TOKENIZER_NAME}' loaded successfully.")
except Exception as e:
    st.error(f"Chyba při načítání tokenizeru '{TOKENIZER_NAME}': {e}")
    logging.error(f"Failed to load tokenizer '{TOKENIZER_NAME}': {e}")

# --- Pomocné Funkce ---

def safe_get_digest(model_data: dict) -> str:
    """Safely extracts the first 6 characters of the model digest."""
    try:
        digest = model_data.get('digest', '')
        if digest and ':' in digest:
            return digest.split(':')[-1][:6]
        return digest[:6] if digest else "latest" # Handle cases without ':' or return first 6 chars
    except Exception as e:
        logging.warning(f"Could not parse digest from model_data: {model_data}. Error: {e}")
        return "unknown"

def load_available_models(host: str) -> list:
    """Fetches available models from the Ollama API."""
    try:
        url = f"{host.strip('/')}{OLLAMA_TAGS_URL}"
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        models_data = response.json().get('models', [])
        # Format: model_name:short_digest (e.g., llama3:8b:1a2b3c)
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
        st.error(f"Chyba při čtení odpovědi od Ollama (neplatný JSON).")
        logging.error(f"Error decoding JSON response from Ollama: {e}")
        return []
    except Exception as e:
        st.error(f"Neočekávaná chyba při načítání modelů: {e}")
        logging.error(f"Unexpected error loading models: {e}")
        return []

# --- Styl ---
# (CSS remains unchanged as requested)
st.markdown("""
<style>
    .user-message {
        background-color: #2e2e2e; padding: 0.75rem 1rem; border-radius: 1rem;
        border: 1px solid #444; margin-top: 0.5rem; width: fit-content;
        max-width: 75%; word-wrap: break-word; line-height: 1.5;
        margin-left: auto; margin-right: 0; display: block; text-align: right; /* Text left aligned, bubble right */
    }
    .bot-message {
        background-color: #1e1e1e; padding: 0.75rem 1rem; border-radius: 1rem;
        border: 1px solid #444; margin-top: 0.5rem; width: fit-content;
        max-width: 75%; word-wrap: break-word; line-height: 1.5;
        margin-left: 0; margin-right: auto; display: block; text-align: left;
    }
    .user-message strong { color: #fdd835; /* zlatá */ }
    .bot-message strong { color: #03dac6; /* tyrkysová */ }
    .stMarkdown { padding: 0; margin: 0; }
    [data-testid="stSidebar"] { background-color: #111111 !important; }
    .stTextArea textarea { background-color: #1e1e1e !important; color: #ffffff !important; }
    .chat-container { max-height: 65vh; overflow-y: auto; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- Session state Inicializace ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "" # Start empty, select default later
if "system_info" not in st.session_state:
    st.session_state.system_info = ""
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = DEFAULT_OLLAMA_HOST

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Nastavení")

    # Input for Ollama Host
    st.session_state.ollama_host = st.text_input(
        "🔗 Ollama API adresa",
        value=st.session_state.ollama_host,
        help="Adresa běžící Ollama instance (např. http://localhost:11434)"
    )

    # Load models dynamically
    available_models = load_available_models(st.session_state.ollama_host)
    
    # Fallback if no models loaded
    if not available_models:
        available_models = ["Nebyly nalezeny modely"] # Placeholder
        st.warning("⚠️ Nelze načíst modely z Ollama. Zkontrolujte adresu a zda Ollama běží.")
        selected_model_index = 0
        st.session_state.selected_model = "" # Clear selection if models fail
    else:
        # Determine default selection index
        if st.session_state.selected_model in available_models:
            selected_model_index = available_models.index(st.session_state.selected_model)
        elif available_models:
             selected_model_index = 0 # Default to the first model if previous selection invalid or empty
             st.session_state.selected_model = available_models[0] # Update session state if defaulting
        else:
            selected_model_index = 0 # Should not happen if available_models has items, but safety first

    # Model Selection Dropdown
    st.session_state.selected_model = st.selectbox(
        "🧠 Vyber model",
        options=available_models,
        index=selected_model_index,
        help="Modely dostupné na vaší Ollama instanci. Formát: název:digest"
    )

    # Model Parameters
    temperature = st.slider("🌡️ Teplota (kreativita)", 0.0, 2.0, 0.7, 0.1, help="Nižší hodnota = více deterministické, vyšší = kreativnější.")
    max_tokens = st.slider("🔢 Max. tokenů v odpovědi", 50, 4096, 512, 50, help="Maximální délka generované odpovědi.")

    # Advanced Settings Expander
    with st.expander("⚙️ Pokročilé nastavení"):
        cz_mode = st.checkbox("🇨🇿 Vynutit češtinu", value=False, help="Přidá instrukci modelu, aby odpovídal česky.")
        stream_mode = st.checkbox("🌀 Streamování odpovědí", value=True, help="Zobrazuje odpověď postupně, jak je generována.")
        show_tokens = st.checkbox("🔢 Zobrazovat tokeny", value=True, help="Zobrazí přibližný počet tokenů v odpovědi.")

    
    # 🔘 Ruční uložení paměti
    if st.sidebar.button("🧠 Uložit paměť ručně"):
        memory_summary = generate_combined_memory(st.session_state.chat_history, tokenizer)
        st.session_state.memory_summary = memory_summary
        st.success("🧠 Shrnutí uloženo do paměti")
    st.markdown("---")

    if st.button("❌ Vymazat historii chatu"):
        st.session_state.chat_history = []
        st.rerun() # Refresh the page to show the cleared history

# --- Hlavní část stránky ---
st.title("🤖 Lokální LLM Chat")
st.caption(f"Powered by Ollama & Streamlit | Model: {st.session_state.selected_model or 'Není vybrán'}")


# --- PLUGINY: Wikipedia, Soubory, Web ---
# Initialize context variables for this run
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
            wikipedia.set_lang("cs")  # nastavíme jazyk pro vyhledávání
            # auto_suggest=True doplňuje neúplné dotazy
            summary = wikipedia.summary(search_term, sentences=3, auto_suggest=True)
            st.markdown("#### Shrnutí článku")
            st.markdown(f"""
                <div style="font-size: 0.85rem; line-height: 1.4; color: #cccccc; background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; margin-top: 0rem;">
                    {summary}
                </div> """, unsafe_allow_html=True)
            st.write("")
            include_wiki = st.checkbox("📎 Připojit shrnutí k dalšímu promptu?", key="wiki_include")
            if include_wiki:
                wiki_context = summary
        except wikipedia.exceptions.PageError:
            st.error(f"❌ Stránka '{search_term}' na Wikipedii nalezena.")
        except wikipedia.exceptions.DisambiguationError as e:
            st.warning(f"🔀 '{search_term}' má více významů. Zkuste upřesnit. Možnosti: {e.options[:5]}")
        except Exception as e:
            st.error(f"❌ Neočekávaná chyba při hledání na Wikipedii: {e}")
            logging.error(f"Wikipedia search error for '{search_term}': {e}")


# 📁 File Upload Plugin
with tab3:
    st.subheader("📂 Nahrát soubor pro kontext")
    uploaded_file = st.file_uploader(
        "Vyber soubor (TXT, PDF, DOCX, CSV)",
        type=["txt", "pdf", "docx", "csv"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            file_type = uploaded_file.name.split(".")[-1].lower()
            bytes_data = uploaded_file.getvalue()  # Read file bytes

            if file_type == "txt":
                # Try common encodings
                try:
                    file_context = bytes_data.decode("utf-8")
                except UnicodeDecodeError:
                    file_context = bytes_data.decode("latin-1")  # Fallback encoding
            elif file_type == "pdf":
                with fitz.open(stream=bytes_data, filetype="pdf") as doc:
                    file_context = "\n".join([page.get_text() for page in doc])
            elif file_type == "docx":
                # Use io.BytesIO to read from memory
                doc = docx.Document(io.BytesIO(bytes_data))
                file_context = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            elif file_type == "csv":
                # Use io.BytesIO for pandas
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
            # Add headers to mimic a browser
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            page = requests.get(web_url, timeout=15, headers=headers)
            page.raise_for_status() # Check for HTTP errors

            # Check content type to avoid parsing non-html
            if 'text/html' in page.headers.get('Content-Type', ''):
                soup = BeautifulSoup(page.content, "html.parser")
                
                # Improved text extraction: Prioritize main content tags, remove script/style
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose() # Remove these tags

                # Get text from meaningful tags, join with newline
                texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'article', 'section'])
                extracted_text = "\n".join(t.get_text(separator=' ', strip=True) for t in texts if t.get_text(strip=True))

                if extracted_text.strip():
                    include_web = st.checkbox("📎 Připojit text z webu k dalšímu promptu?", key="web_include")
                    if include_web:
                        web_context = extracted_text # Use full extracted text
                else:
                    st.warning("⚠️ Nepodařilo se extrahovat žádný čitelný text z hlavních tagů stránky.")
            else:
                st.warning(f"⚠️ Obsah na URL není HTML (Content-Type: {page.headers.get('Content-Type', 'N/A')}). Nelze extrahovat text.")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Chyba při načítání URL: {e}")
            logging.error(f"Error fetching URL '{web_url}': {e}")
        except Exception as e:
            st.error(f"❌ Neočekávaná chyba při zpracování webu: {e}")
            logging.error(f"Error processing web content from '{web_url}': {e}\n{traceback.format_exc()}")

# --- Zpracování Chatu ---

# Vstupní formulář
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("💬 Napiš dotaz:", height=100, placeholder="Zadej svůj dotaz nebo instrukci...", key="user_input")
    submitted = st.form_submit_button("🚀 Odeslat")

# Pokud byl formulář odeslán a vstup není prázdný
if submitted and user_input.strip() and st.session_state.selected_model and "Nebyly nalezeny modely" not in st.session_state.selected_model:

    # 1. Sestavení historie pro kontext
    # Only include user/bot turns for context history passed to model
    
# 🔁 Vložení shrnutí paměti (pokud existuje)
memory_summary = st.session_state.get("memory_summary", "")
if memory_summary:
    context_history.insert(0, {"role": "system", "content": memory_summary})
context_history = []

    if cz_mode:  # 🇨🇿 checkbox z GUI
        context_history.append({
            "role": "system",
            "content": "Odpovídej výhradně česky. Nemusíš zmiňovat, že odpovídáš česky."
        })

    # přidání historie
    for sender, message, _ in st.session_state.chat_history:
         # Simple format for model context
         role = "user" if sender == "user" else "assistant"
         # Extract only the core message, removing potential token counts etc.
         core_message = message.split("\n\n*Přibližný počet tokenů:")[0].strip()
         context_history.append({"role": role, "content": core_message})


    # 2. Příprava aktuálního promptu
    final_prompt = user_input.strip()
    
    # Prepend context from plugins if included
    prepended_context = ""
    if include_wiki and wiki_context:
        prepended_context += f"[WIKIPEDIA SHRNUTÍ]:\n{wiki_context}\n\n"
    if include_file and file_context:
         # Limit context size reasonably if needed, e.g., first/last N chars/tokens
         # For now, using full context but be mindful of model limits
        prepended_context += f"[OBSAH SOUBORU]:\n{file_context[:10000]}\n\n" # Example limit
    if include_web and web_context:
        prepended_context += f"[TEXT Z WEBU]:\n{web_context[:10000]}\n\n" # Example limit
        
    if prepended_context:
        final_prompt = f"{prepended_context}Můj dotaz s využitím předchozího kontextu:\n{final_prompt}"

    # 3. Přidání uživatelského vstupu do historie chatu (pro zobrazení)
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Store the original user input for display, not necessarily the one sent to model if context was added
    st.session_state.chat_history.append(("user", user_input.strip(), timestamp)) 
    
    # 4. Příprava payloadu pro Ollama API

    # Get the full selected model string (e.g., "mistral-nemo:12b:994f3b" or "llama2:latest")
    selected_model_full = st.session_state.selected_model

    # Prepare the model name for the API: needs to be "name:tag" format.
    # Strip the digest part if it exists (the last part after a colon).
    parts = selected_model_full.split(':')
    if len(parts) > 2:  # If format is name:tag:digest
        model_name_for_api = f"{parts[0]}:{parts[1]}" # Combine only name and tag
    else: # If format is already name:tag or just name (implicitly latest)
        model_name_for_api = selected_model_full # Use as is

    # Optional: Add a check here if needed, but the outer check might suffice
    if not model_name_for_api or "Nebyly nalezeny modely" in model_name_for_api:
         st.warning("⚠️ Prosím, vyberte platný model v nastavení.")
         # Potentially add st.stop() or return here if you want extra safety
    else:
        logging.info(f"Selected model (full): {selected_model_full}, Sending to API as: {model_name_for_api}") # Log for verification
        
        # Prepare the payload for the API call
        final_answer = ""

        # 1. ZKUSÍME messages
        context_history = []
        
        if cz_mode:
            context_history.append({
                "role": "system",
                "content": "Odpovídej výhradně v českém jazyce. Nemusíš zmiňovat, že odpovídáš česky."
            })
        
        # Přidej historii do kontextu
        for sender, message, _ in st.session_state.chat_history:
            role = "user" if sender == "user" else "assistant"
            cleaned = message.split("\n\n*Přibližný počet tokenů:")[0].strip()
            context_history.append({"role": role, "content": cleaned})
        
        # Přidej aktuální prompt
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
                response = requests.post(
                    f"{st.session_state.ollama_host}{OLLAMA_GENERATE_URL}",
                    json=payload,
                    stream=stream_mode,
                    timeout=120
                )
        
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
        
        # 2. FALLBACK na prompt, pokud výsledek je prázdný
        if not final_answer:
            fallback_prompt = final_prompt
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
                    response = requests.post(
                        f"{st.session_state.ollama_host}{OLLAMA_GENERATE_URL}",
                        json=payload,
                        stream=stream_mode,
                        timeout=120
                    )
        
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
        
            except Exception as e:
                st.error("❌ Chyba i při fallback režimu.")

            except requests.exceptions.Timeout:
                st.error("❌ Vypršel časový limit při čekání na odpověď od Ollama.")
                logging.error("Ollama API request timed out.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Chyba komunikace s Ollama API: {e}")
                logging.error(f"Ollama API request error: {e}\n{traceback.format_exc()}")
            except Exception as e:
                st.error(f"⚠️ Neočekávaná chyba při generování odpovědi: {e}")
                logging.error(f"Unexpected error during Ollama call: {e}\n{traceback.format_exc()}")
                # Removed the user message if bot failed, or keep it? Let's keep it for context.
                # st.session_state.chat_history.pop() # Remove user message if bot fails?
        
        # 3. ULOŽ odpověď, pokud nějaká existuje
        if final_answer:
            token_count = len(tokenizer.encode(final_answer)) if tokenizer else len(final_answer.split())
            if show_tokens:
                final_answer += f"\n\n*Přibližný počet tokenů: {token_count}*"
            st.session_state.chat_history.append(("bot", final_answer, datetime.now().strftime("%H:%M:%S")))
            st.rerun()
        
        else:
             st.warning("⚠️ Model vrátil prázdnou odpověď.")

elif submitted and (not user_input.strip()):
     st.warning("⚠️ Prosím, zadejte dotaz.")
elif submitted and (not st.session_state.selected_model or "Nebyly nalezeny modely" in st.session_state.selected_model):
     st.warning("⚠️ Prosím, vyberte platný model v nastavení.")

# --- Zobrazení historie chatu ---
st.markdown("### Historie konverzace")
st.markdown('<div id="chat-container" class="chat-container">', unsafe_allow_html=True)

# Display chat history in reverse order (newest first)
if st.session_state.chat_history:
    # Iterate in reverse for display
    for sender, message, timestamp in reversed(st.session_state.chat_history):
        css_class = "user-message" if sender == "user" else "bot-message"
        icon = '🧑' if sender == 'user' else '🤖'
        sender_name = 'Ty' if sender == 'user' else 'LLM'

        # Use st.markdown for rendering, including potential markdown in messages
        # CORRECTED LINE: Removed the invalid comment inside the f-string
        formatted_message = f"""
        <div class="{css_class}">
            <strong>{icon} {sender_name}</strong>
            <div style="margin-top: 0.3rem;">{message.replace('<', '&lt;').replace('>', '&gt;')}</div>
            <div style="font-size: 0.8rem; color: #888; text-align: right; margin-top: 0.5rem;">{timestamp}</div>
        </div>
        """
        st.markdown(formatted_message, unsafe_allow_html=True)
        # Add a small vertical space between messages
        st.write("") # Adds a controlled small space

st.markdown('</div>', unsafe_allow_html=True)


# --- Získání systémových informací (jednou) ---
if not st.session_state.system_info:
    try:
        version_url = f"{st.session_state.ollama_host.strip('/')}{OLLAMA_VERSION_URL}"
        system_response = requests.get(version_url, timeout=5)
        if system_response.status_code == 200:
            version_info = system_response.json()
            st.session_state.system_info = f"Ollama verze: {version_info.get('version', 'N/A')}"
            logging.info(f"Ollama version check successful: {st.session_state.system_info}")
            # Display in sidebar footer or somewhere less prominent
            st.sidebar.caption(st.session_state.system_info)
        else:
            logging.warning(f"Ollama version check failed with status: {system_response.status_code}")
            st.sidebar.caption("⚠️ Nelze ověřit verzi Ollama.")
    except Exception as e:
        logging.warning(f"Failed to get Ollama version info: {e}")
        st.sidebar.caption("⚠️ Nelze se připojit k Ollama pro info.")

# --- END OF FILE chat_llm.py --
    # Přidáme paměť i do fallback promptu
    memory_summary = st.session_state.get("memory_summary", "")
    if memory_summary:
        fallback_prompt = memory_summary + "\n\n" + fallback_prompt