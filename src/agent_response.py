import os
import time
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from src.data_loader import load_chat_data, load_policy_docs
from src.embedding_index import build_or_load_index

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_last_call_ts = 0.0
_min_call_interval = float(os.getenv("MIN_LLM_INTERVAL", 1.0))  # seconds

# Prompt template for LLM decision inference
DECISION_PROMPT_TEMPLATE = """
You are an AI underwriting assistant for small business insurance.
You have access to the following context from previous broker chats and policy documents:
{context}

Given this information and the broker's question:
{question}

Please provide:
1) A clear, concise answer to the question.
2) A recommended underwriting decision: Accept, Decline, Refer, or Discount.
3) A short explanation for your decision.

Format your response as:

Answer:
Decision:
Explanation:
"""

def chat_with_agent(query):
    global _last_call_ts
    now = time.time()
    if now - _last_call_ts < _min_call_interval:
        time.sleep(_min_call_interval - (now - _last_call_ts))
    _last_call_ts = time.time()
    # Load chat history data from Excel file
    chat_data = load_chat_data("data/chat_data/chat_data.xlsx")
    print(f"[DEBUG] Loaded {len(chat_data)} chat documents")  # Debug: Number of chat docs loaded

    # Load policy documents from the specified folder
    policy_docs = load_policy_docs("data/policy_documents")
    print(f"[DEBUG] Loaded {len(policy_docs)} policy documents")  # Debug: Number of policy docs loaded

    # Build or load the FAISS vector index using chat and policy docs
    index = build_or_load_index(chat_data, policy_docs)
    print("[DEBUG] Index built or loaded successfully")  # Debug: Index status

    # Initialize the OpenAI chat model with deterministic output (temperature=0)
    llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME)
    print("[DEBUG] ChatOpenAI model initialized")  # Debug: LLM status

    # Create a retriever from the index for semantic search
    retriever = index.as_retriever()
    print("[DEBUG] Retriever created from index")  # Debug: Retriever status

    # Retrieve relevant documents from the index based on the user's query
    retrieved_docs = retriever.get_relevant_documents(query)
    print(f"[DEBUG] Retrieved {len(retrieved_docs)} relevant documents for query")  # Debug: Retrieval count

    # Concatenate the content of retrieved documents to form the context for the prompt
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    print(f"[DEBUG] Context for prompt (first 500 chars):\n{context[:500]}")  # Debug: Context preview

    # Format the prompt by injecting the context and the user's question
    prompt = DECISION_PROMPT_TEMPLATE.format(context=context, question=query)
    print(f"[DEBUG] Prompt for LLM (first 500 chars):\n{prompt[:500]}")  # Debug: Prompt preview

    # Query the LLM with the full prompt and get the response
    try:
        response = llm.call_as_llm(prompt)
    except Exception as e:
        response = f"Error contacting LLM: {e}"
    print(f"[DEBUG] LLM response:\n{response}")  # Debug: LLM output

    # Return the model's response (should include Answer, Decision, Explanation)
    return response