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
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

_vectorstore = None
FAISS_DB_PATH = "vector_db"   # local directory for FAISS index

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

    # Initialize embeddings
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = _get_api_key()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=embedding_model)
    print(f"[DEBUG] Using embedding model: {embedding_model}")

    # Attempt Chroma load/create unless FAISS forced
    # Load FAISS if present
    if os.path.exists(FAISS_DB_PATH) and os.listdir(FAISS_DB_PATH):
        try:
            _vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            print(f"[DEBUG] FAISS index loaded from {FAISS_DB_PATH}")
            return _vectorstore
        except Exception as e:
            print(f"[WARN] Failed to load existing FAISS index ({e}); rebuilding.")

    # Wrap chat_data_chunks if needed
    if isinstance(chat_data_chunks[0], Document):
        chat_docs = chat_data_chunks
        print("[DEBUG] chat_data_chunks are already Document objects")
    else:
        chat_docs = [Document(page_content=chunk) for chunk in chat_data_chunks]
        print(f"[DEBUG] Wrapped {len(chat_docs)} chat chunks into Document objects")

    all_docs = chat_docs + policy_docs
    print(f"[DEBUG] Total documents for indexing: {len(all_docs)}")

    # Build FAISS from documents
    _vectorstore = FAISS.from_documents(all_docs, embeddings)
    print("[DEBUG] FAISS vectorstore created from documents")
    try:
        os.makedirs(FAISS_DB_PATH, exist_ok=True)
        _vectorstore.save_local(FAISS_DB_PATH)
        print(f"[DEBUG] FAISS index saved to {FAISS_DB_PATH}")
    except Exception as e:
        print(f"[WARN] Could not persist FAISS index: {e}")

    return _vectorstore
