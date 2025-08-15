import os
import time
import re
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
from src.data_loader import load_chat_data, load_policy_docs
from src.embedding_index import build_or_load_index

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_last_call_ts = 0.0
_min_call_interval = float(os.getenv("MIN_LLM_INTERVAL", 1.0))  # seconds

# Core underwriting prompt (memory + retrieved context + current question)
AGENT_NAME = os.getenv("AGENT_NAME", "Agenverse")
DECISION_PROMPT_TEMPLATE = """
You are an experienced, professional yet approachable commercial SME underwriting assistant. Your name is {agent_name}. Always state your name as {agent_name} if the broker asks who you are or your name.
Goals:
1. Be concise, clear, human and empathetic (avoid sounding robotic; use natural wording, no over-formality).
2. Use prior conversation context when relevant; don't repeat what the broker already knows unless clarifying.
3. Always separate a direct Answer from the underwriting Decision and an Explanation.

Conversation history (most recent first):
{history}

Retrieved reference material (may be partial snippets from chats / policy docs):
{context}

Current broker question:
{question}

If the question is OUT OF SCOPE of underwriting or lacks sufficient detail, first briefly acknowledge and ask a focused clarification question (one sentence) before attempting a decision; if truly unrelated (pure small talk), politely switch to a friendly general reply and skip Decision/Explanation.

Otherwise provide strictly this format (omit brackets):
Answer: <concise direct answer; if assumptions made, state them>
Decision: <Accept | Decline | Refer | Discount | Need Clarification>
Explanation: <key reasoning factors referencing risk, coverage, limits, exclusions, appetite>
"""

SMALL_TALK_PATTERNS = [
    r"^hi$", r"^hi there", r"^hello", r"hey", r"how are you", r"thank(s| you)", r"good (morning|afternoon|evening)",
    r"what's up", r"how's it going", r"who are you", r"help$", r"^yo$"
]

GENERAL_RESPONSE_TEMPLATES = [
    f"Hi there! I'm {AGENT_NAME}. How can I help with an underwriting or coverage question today?",
    f"Hello! I'm {AGENT_NAME}, here to help. Let me know any risk details or policy points you're exploring.",
    f"I'm {AGENT_NAME}. I can assist with underwriting decisions, coverage clarifications, or policy wording questions—what would you like to dive into?",
    f"Thanks for reaching out—{AGENT_NAME} at your service. Share the business type, key exposures, or your specific question and I'll help." 
]

OUT_OF_SCOPE_KEYWORDS = ["weather", "movie", "music", "recipe", "game", "restaurant", "travel plan"]

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 8))  # number of previous exchanges (user+agent pairs) to include

def _is_name_query(query: str) -> bool:
    q = query.lower().strip()
    triggers = ["your name", "who are you", "what are you", "what's your name", "who is this", "who am i chatting with", "agent name"]
    return any(t in q for t in triggers)

PERSONAL_EMOTION_WORDS = [
    "sad","upset","down","depressed","happy","angry","anxious","stressed","tired","lonely","excited","frustrated","worried","confused"
]

def _is_personal(query: str) -> bool:
    q = query.lower().strip()
    # First-person + emotion word heuristic
    if any(pron in q for pron in ["i am ", "i'm ", "i feel", "feeling "]):
        if any(w in q for w in PERSONAL_EMOTION_WORDS):
            return True
    # Direct short emotional statements
    if q in [f"i am {w}" for w in PERSONAL_EMOTION_WORDS]:
        return True
    return False

def _is_small_talk(query: str) -> bool:
    q = query.lower().strip()
    for pat in SMALL_TALK_PATTERNS:
        if re.search(pat, q):
            return True
    # Very short single-word or greeting-like
    if len(q.split()) <= 3 and all(len(w) <= 7 for w in q.split()):
        return True
    return False

def _is_out_of_scope(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in OUT_OF_SCOPE_KEYWORDS)

def _build_history_block(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "(no prior conversation)"
    # Take last N turns (a turn = user+agent message). We'll just slice last 2*N messages.
    trimmed = history[-(MAX_HISTORY_TURNS*2):]
    lines = []
    for speaker, msg in trimmed[::-1]:  # reverse chronological as template says most recent first
        # Normalize speaker labels
        role = "Broker" if "Broker" in speaker or speaker.startswith("🧑") else ("Agent" if "Agent" in speaker else speaker)
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)

def _general_response(query: str, history: List[Tuple[str, str]]) -> str:
    # Provide a friendly response & gentle steer back to underwriting domain.
    import random
    base = random.choice(GENERAL_RESPONSE_TEMPLATES)
    if _is_name_query(query):
        return f"Answer: I'm {AGENT_NAME}, your commercial SME underwriting assistant.\nDecision: Need Clarification\nExplanation: Share the risk type, key exposures, or coverage question so I can help further."
    if _is_personal(query):
        # Empathetic single-line (no Decision/Explanation for personal/emotional statements)
        return "I'm here to help. If you have any underwriting or coverage questions, feel free to share them when you're ready." if len(query.split()) <= 6 else "I hear you. Let me know any insurance or coverage concerns and I'll support you—no rush." 
    if _is_small_talk(query):
        return base
    if _is_out_of_scope(query):
        return base + " It seems your last question isn't underwriting related, but feel free to ask anything about coverage, appetite, pricing or broker interactions." 
    return base

def _normalize_section_formatting(text: str) -> str:
    """Ensure 'Answer:', 'Decision:', 'Explanation:' each start on their own line.
    Also strip extra spaces after the labels.
    """
    # Insert newline before labels if missing (except if already at start)
    for label in ["Decision:", "Explanation:"]:
        # If label appears after other text on same line, force newline before it
        text = re.sub(rf"\s*{label}", f"\n{label}", text)
    # Ensure order lines are separated
    # Collapse multiple spaces after label
    text = re.sub(r"(Answer:|Decision:|Explanation:)\s+", r"\1 ", text)
    # Guarantee final newline separation (in case LLM jammed them together)
    # If Explanation immediately follows Decision on same line, we already forced newline before Explanation
    return text.strip()

def chat_with_agent(query: str, history: Optional[List[Tuple[str, str]]] = None):
    global _last_call_ts
    now = time.time()
    if now - _last_call_ts < _min_call_interval:
        time.sleep(_min_call_interval - (now - _last_call_ts))
    _last_call_ts = time.time()
    history = history or []

    # Early small-talk / out-of-scope detection (fast path before retrieval cost)
    if (_is_small_talk(query) and len(query.split()) < 10) or _is_personal(query):
        return _general_response(query, history)
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

    # Similarity search with scores to decide if retrieval is meaningful
    try:
        retrieved_with_scores = index.similarity_search_with_score(query, k=4)
        retrieved_docs = [d for d, _ in retrieved_with_scores]
        scores = [s for _, s in retrieved_with_scores]
        print(f"[DEBUG] Retrieved {len(retrieved_docs)} docs with scores: {scores}")
    except Exception as e:
        print(f"[WARN] similarity_search_with_score failed ({e}); falling back to retriever")
        retriever = index.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(query)
        scores = []
    # Heuristic: if no docs or scores indicate weak match (FAISS L2 distance large)
    weak_context = False
    if not retrieved_docs:
        weak_context = True
    elif scores:
        # Distances vary; we'll use a very rough heuristic threshold
        avg_score = sum(scores)/len(scores)
        weak_context = avg_score > 1.5  # tune empirically later
        print(f"[DEBUG] avg_score={avg_score} weak_context={weak_context}")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    history_block = _build_history_block(history)

    # Out-of-scope fallback if retrieval weak AND query looks unrelated
    if weak_context and (_is_out_of_scope(query) or len(query.split()) < 4):
        return _general_response(query, history)

    prompt = DECISION_PROMPT_TEMPLATE.format(context=context or "(no strong matches)", history=history_block, question=query, agent_name=AGENT_NAME)
    print(f"[DEBUG] Prompt for LLM (first 500 chars):\n{prompt[:500]}")  # Debug: Prompt preview

    # Query the LLM with the full prompt and get the response
    try:
        response = llm.call_as_llm(prompt)
    except Exception as e:
        response = _general_response(query, history) + f" (Encountered an error reaching model: {e})"
    print(f"[DEBUG] LLM response:\n{response}")  # Debug: LLM output

    # Post-process: if model forgot required sections and it's an underwriting-style query, patch minimally
    if "Decision:" not in response and not _is_small_talk(query) and not _is_out_of_scope(query) and not _is_personal(query):
        response += "\nDecision: Need Clarification\nExplanation: I didn't receive enough structured info; please provide key risk details (trade, sums insured, claims history)."
    # Normalize formatting (each section its own line)
    response = _normalize_section_formatting(response)
    return response