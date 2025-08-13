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
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

_vectorstore = None
VECTOR_DB_PATH = "chroma_db"  # folder, not a single file

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def build_or_load_index(chat_data_chunks, policy_docs):
    global _vectorstore

    # If the vectorstore is already loaded in memory, return it
    if _vectorstore is not None:
        print("[DEBUG] Returning in-memory vectorstore")
        return _vectorstore

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # If Chroma index exists on disk, load it
    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        print(f"[DEBUG] Loading Chroma vectorstore from {VECTOR_DB_PATH}")
        _vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        print("[DEBUG] Chroma vectorstore loaded from disk")
        return _vectorstore

    # Wrap chat_data_chunks if needed
    if isinstance(chat_data_chunks[0], Document):
        chat_docs = chat_data_chunks
        print("[DEBUG] chat_data_chunks are already Document objects")
    else:
        chat_docs = [Document(page_content=chunk) for chunk in chat_data_chunks]
        print(f"[DEBUG] Wrapped {len(chat_docs)} chat chunks into Document objects")

    all_docs = chat_docs + policy_docs
    print(f"[DEBUG] Total documents for indexing: {len(all_docs)}")

    # Create new Chroma vectorstore from documents
    _vectorstore = Chroma.from_documents(all_docs, embeddings, persist_directory=VECTOR_DB_PATH)
    print("[DEBUG] Chroma vectorstore created from documents")

    # Persist vectorstore to disk
    _vectorstore.persist()
    print(f"[DEBUG] Chroma vectorstore persisted to {VECTOR_DB_PATH}")

    return _vectorstore
