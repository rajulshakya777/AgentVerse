import os
import streamlit as st
from dotenv import load_dotenv
from src.agent_response import chat_with_agent
from src.data_loader import load_chat_data, load_policy_docs
from src.embedding_index import build_or_load_index

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Safe Streamlit secrets access (avoid exception if no secrets file locally)
if not os.getenv("OPENAI_API_KEY"):
    try:
        secret_key = None
        # st.secrets may raise if no secrets config; guard with try
        try:
            secret_key = st.secrets.get("OPENAI_API_KEY")  # returns None if missing
        except Exception:
            secret_key = None
        if secret_key:
            os.environ["OPENAI_API_KEY"] = secret_key
    except Exception:
        pass

api_key_present = bool(os.getenv("OPENAI_API_KEY"))
if not api_key_present:
    st.warning("OPENAI_API_KEY not set. Add it as a Streamlit secret or environment variable.")

# Set Streamlit app title
st.set_page_config(page_title="Underwriting AI Agent", page_icon="🛡️")
st.title("🛡️ Underwriting AI Agent for Brokers")
st.caption("Ask underwriting questions. The system references broker chats + policy docs.")

@st.cache_resource(show_spinner="Building vector index (first run)...")
def get_index():
    chat_data = load_chat_data("data/chat_data/chat_data.xlsx")
    policy_docs = load_policy_docs("data/policy_documents")
    return build_or_load_index(chat_data, policy_docs)

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    print("[DEBUG] Initialized chat_history in session state")  # Debug: Session state init

# Input box for user to ask a question
user_input = st.text_input("Ask a question about underwriting, coverage, or policy:")
debug = st.sidebar.checkbox("Show debug traces", value=False)
if user_input:
    st.session_state.chat_history.append(("🧑 Broker", user_input))
    with st.spinner("Thinking..."):
        try:
            _ = get_index()  # build / load once
            response = chat_with_agent(user_input)
        except Exception as e:
            import traceback, textwrap
            tb = traceback.format_exc()
            msg = f"Error: {e}"
            st.error(msg)
            if debug:
                st.code(tb, language="text")
            response = msg
        finally:
            st.session_state.chat_history.append(("🤖 Agent", response))

# Display the chat history
with st.container():
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")

st.sidebar.header("Settings")
st.sidebar.write("Adjust chunking via env vars: CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS.")
st.sidebar.write("Model: "+ os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))