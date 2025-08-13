import os
import os
import hashlib
import pandas as pd
from langchain.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

# Size-reduction configuration (can be overridden via env vars)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))            # Larger chunks = fewer vectors
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))        # Smaller overlap reduces duplication
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", 80))    # Drop trivial / boilerplate chunks

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def dedupe_documents(documents):
    """Remove exact duplicate (normalized) page_content documents to cut index size."""
    seen = set()
    unique = []
    for d in documents:
        norm = _normalize(d.page_content)
        h = _hash(norm)
        if h in seen:
            continue
        seen.add(h)
        unique.append(d)
    return unique

def chunk_text(text, source=""):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for chunk in splitter.split_text(text):
        if len(chunk) < MIN_CHUNK_CHARS:
            continue  # skip very small fragments
        chunks.append(Document(page_content=chunk, metadata={"source": source}))
    return chunks

def load_chat_data(chat_excel_path):
    df = pd.read_excel(chat_excel_path)
    print(f"[DEBUG] Loaded chat Excel file: {chat_excel_path} with {len(df)} rows")
    all_docs = []

    for _, row in df.iterrows():
        transcript = row.get('TRANSCRIPT', '')
        if pd.isna(transcript):
            continue

        transcript_clean = re.sub(r'<.*?>', '', str(transcript))
        lines = transcript_clean.strip().split('\n')
        messages = []

        for line in lines:
            match = re.match(r'\d{2}:\d{2}:\d{2} - (.*?) - (.*)', line)
            if match:
                sender, message = match.groups()
                if sender.strip():
                    messages.append(f"{sender.strip()}: {message.strip()}")

        chat_text = "\n".join(messages)
        metadata = {
            "experience": row.get("EXPERIENCE", ""),
            "initial_group": row.get("INITIAL ROUTING GROUP", ""),
            "final_group": row.get("FINAL ROUTING GROUP", ""),
            "outcome": row.get("OUTCOME", "")
        }

        # Chunk and attach minimal metadata (avoid storing large redundant keys)
        chunks = chunk_text(chat_text, source="chat")
        for chunk in chunks:
            # Only keep small essential metadata keys
            for k, v in metadata.items():
                if v:  # don't store empty strings
                    chunk.metadata[k] = v
            all_docs.append(chunk)

    # Deduplicate after chunking to remove overlapping / repeated content
    before = len(all_docs)
    all_docs = dedupe_documents(all_docs)
    after = len(all_docs)
    print(f"[DEBUG] Returning {after} (was {before}) deduped chat Document objects from file: {chat_excel_path}")
    return all_docs

def load_policy_docs(policy_folder):
    documents = []
    print(f"[DEBUG] Scanning policy folder: {policy_folder}")
    for root, _, files in os.walk(policy_folder):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(path)
                print(f"[DEBUG] Loading PDF: {path}")
            elif file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
                print(f"[DEBUG] Loading DOCX: {path}")
            else:
                print(f"[DEBUG] Skipping unsupported file: {path}")
                continue
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            # Basic boilerplate filter (drop very short whole-doc extracts)
            chunks = chunk_text(text, source=file)
            documents.extend(chunks)
    before = len(documents)
    documents = dedupe_documents(documents)
    after = len(documents)
    print(f"[DEBUG] Returning {after} (was {before}) policy Document objects from folder: {policy_folder}")
    return documents
