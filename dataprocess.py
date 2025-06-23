import pandas as pd
import re
import os
from langchain.schema import Document


from langchain_core.documents import Document
import pandas as pd
import re

def load_chat_data(chat_excel_path):
    df = pd.read_excel(chat_excel_path)
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

        all_docs.append(Document(page_content=chat_text, metadata=metadata))

    return all_docs


# Test it independently
def test_load_chat_data():
    test_path = "data/chat_data/chat_data.xlsx"  # adjust path if needed

    if not os.path.exists(test_path):
        print(f"File not found: {test_path}")
        return

    documents = load_chat_data(test_path)

    assert isinstance(documents, list), "Returned value should be a list."
    assert all(isinstance(doc, Document) for doc in documents), "All elements should be Document instances."

    # Print first few entries for visual inspection
    for i, doc in enumerate(documents[:3]):
        print(f"\n--- Document {i+1} ---")
        print("Content:\n", doc.page_content[:500], "...\n")  # Print only the first 500 chars
        print("Metadata:", doc.metadata)

    print("\n✅ Test passed: All documents loaded correctly.")

if __name__ == "__main__":
    test_load_chat_data()

