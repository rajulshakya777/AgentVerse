# AI Underwriting Agent MVP

A local MVP prototype of an AI-powered assistant that simulates underwriting decisions based on historical chat data and policy rules.

## Features
- Trained on past chat conversations and underwriting policy documents
- Makes real-time decisions: accept, decline, refer, discount, coverage check
- Simple web chat interface using Streamlit

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
underwriting_agent_mvp/
├── app.py
├── data/
│   └── chat_data/
│       └── combined_chat_data.xlsx
│   └── policy_documents/
│       └── [business_type]/policy1.pdf, policy2.docx...
├── src/
│   └── data_loader.py
│   └── embedding_index.py
│   └── inference_engine.py
│   └── agent_response.py
├── requirements.txt
├── README.md
```
# AgentVerse
