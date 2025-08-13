# import os
# from dotenv import load_dotenv
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.schema import Document

# _vectorstore = None
# VECTOR_DB_PATH = "vector_db"

# load_dotenv()  # load environment variables from .env file
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# def build_or_load_index(chat_data_chunks, policy_docs):
#     global _vectorstore

#     # If the vectorstore is already loaded in memory, return it
#     if _vectorstore is not None:
#         print("[DEBUG] Returning in-memory vectorstore")  # Debug: Using cached vectorstore
#         return _vectorstore

#     # If the FAISS index exists on disk, load it
#     if os.path.exists(VECTOR_DB_PATH):
#         print(f"[DEBUG] Loading FAISS vectorstore from {VECTOR_DB_PATH}")  # Debug: Loading from disk
#         embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#         _vectorstore = FAISS.load_local(
#             VECTOR_DB_PATH, 
#             embeddings, 
#             allow_dangerous_deserialization=True  # <-- Add this flag
#         )
#         print("[DEBUG] FAISS vectorstore loaded from disk")  # Debug: Loaded from disk
#         return _vectorstore

#     # If not, create new Documents from chat chunks (if not already Documents)
#     # If chat_data_chunks are already Document objects, skip this wrapping!
#     if isinstance(chat_data_chunks[0], Document):
#         chat_docs = chat_data_chunks
#         print("[DEBUG] chat_data_chunks are already Document objects")  # Debug: No wrapping needed
#     else:
#         chat_docs = [Document(page_content=chunk) for chunk in chat_data_chunks]
#         print(f"[DEBUG] Wrapped {len(chat_docs)} chat chunks into Document objects")  # Debug: Wrapping

#     # Combine chat and policy docs
#     all_docs = chat_docs + policy_docs
#     print(f"[DEBUG] Total documents for indexing: {len(all_docs)}")  # Debug: Total docs

#     # Create embeddings and FAISS index
#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     print("[DEBUG] OpenAIEmbeddings initialized")  # Debug: Embeddings status
#     _vectorstore = FAISS.from_documents(all_docs, embeddings)
#     print("[DEBUG] FAISS vectorstore created from documents")  # Debug: Vectorstore creation

#     # Save the FAISS index locally for future use
#     _vectorstore.save_local(VECTOR_DB_PATH)
#     print(f"[DEBUG] FAISS vectorstore saved to {VECTOR_DB_PATH}")  # Debug: Save status

#     return _vectorstore

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings
from langchain.schema import Document

_vectorstore = None
VECTOR_DB_PATH = "chroma_db"  # folder, not a single file
FAISS_DB_PATH = "vector_db"   # fallback path if Chroma not usable

load_dotenv()

def _get_api_key():
    # Re-read each time so late injection (e.g. Streamlit secrets) works
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # Try Streamlit secrets directly (works even if this module imported before app sets env)
        try:
            import streamlit as st  # local import to avoid hard dependency at non-Streamlit runtime
            if "OPENAI_API_KEY" in st.secrets:
                key = st.secrets["OPENAI_API_KEY"]
                os.environ["OPENAI_API_KEY"] = key  # cache for other libs
        except Exception:
            pass
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it as an env var or Streamlit secret (OPENAI_API_KEY)."
        )
    return key

def build_or_load_index(chat_data_chunks, policy_docs):
    global _vectorstore

    # If the vectorstore is already loaded in memory, return it
    if _vectorstore is not None:
        print("[DEBUG] Returning in-memory vectorstore")
        return _vectorstore

    # Smaller model can reduce embedding dimensionality & disk usage
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = _get_api_key()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=embedding_model)
    print(f"[DEBUG] Using embedding model: {embedding_model}")

    prefer_faiss = os.getenv("USE_FAISS_FALLBACK", "0") == "1"
    chroma_impl = os.getenv("CHROMA_DB_IMPL", "duckdb+parquet")  # avoid old sqlite on Streamlit
    client_settings = Settings(chroma_db_impl=chroma_impl, persist_directory=VECTOR_DB_PATH)

    # Attempt Chroma load/create unless FAISS forced
    if not prefer_faiss:
        try:
            if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
                print(f"[DEBUG] Loading Chroma vectorstore from {VECTOR_DB_PATH} using impl={chroma_impl}")
                _vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings, client_settings=client_settings)
                print("[DEBUG] Chroma vectorstore loaded from disk")
                return _vectorstore
        except Exception as e:
            print(f"[WARN] Chroma load failed ({e}); falling back to FAISS.")
            prefer_faiss = True

    # Wrap chat_data_chunks if needed
    if isinstance(chat_data_chunks[0], Document):
        chat_docs = chat_data_chunks
        print("[DEBUG] chat_data_chunks are already Document objects")
    else:
        chat_docs = [Document(page_content=chunk) for chunk in chat_data_chunks]
        print(f"[DEBUG] Wrapped {len(chat_docs)} chat chunks into Document objects")

    all_docs = chat_docs + policy_docs
    print(f"[DEBUG] Total documents for indexing: {len(all_docs)}")

    if not prefer_faiss:
        try:
            _vectorstore = Chroma.from_documents(all_docs, embeddings, persist_directory=VECTOR_DB_PATH, client_settings=client_settings)
            print("[DEBUG] Chroma vectorstore created from documents")
            _vectorstore.persist()
            print(f"[DEBUG] Chroma vectorstore persisted to {VECTOR_DB_PATH}")
            return _vectorstore
        except Exception as e:
            print(f"[WARN] Chroma creation failed ({e}); using FAISS fallback.")
            prefer_faiss = True

    # FAISS fallback (non-persistent unless you implement save/load manually)
    _vectorstore = FAISS.from_documents(all_docs, embeddings)
    print("[DEBUG] FAISS vectorstore created (fallback mode)")
    # Optional: save FAISS index if desired
    try:
        os.makedirs(FAISS_DB_PATH, exist_ok=True)
        _vectorstore.save_local(FAISS_DB_PATH)
        print(f"[DEBUG] FAISS index saved to {FAISS_DB_PATH}")
    except Exception as e:
        print(f"[WARN] Could not persist FAISS index: {e}")

    return _vectorstore
