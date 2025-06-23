import os
import pandas as pd
from langchain.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

def chunk_text(text, source=""):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    return [Document(page_content=chunk, metadata={"source": source}) for chunk in splitter.split_text(text)]

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

        chunks = chunk_text(chat_text, source="chat")
        for chunk in chunks:
            chunk.metadata.update(metadata)
            all_docs.append(chunk)

    print(f"[DEBUG] Returning {len(all_docs)} chat Document objects from file: {chat_excel_path}")
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
            chunks = chunk_text(text, source=file)
            documents.extend(chunks)
    print(f"[DEBUG] Returning {len(documents)} policy Document objects from folder: {policy_folder}")
    return documents
