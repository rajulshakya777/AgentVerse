import streamlit as st
from dotenv import load_dotenv
from src.agent_response import chat_with_agent

# Load environment variables (e.g., OpenAI API key)
load_dotenv()
print("[DEBUG] Environment variables loaded")  # Debug: .env loaded

# Set Streamlit app title
st.title("🛡️ Underwriting AI Agent for Brokers")

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    print("[DEBUG] Initialized chat_history in session state")  # Debug: Session state init

# Input box for user to ask a question
user_input = st.text_input("Ask a question about underwriting, coverage, or policy:")
if user_input:
    print(f"[DEBUG] User input received: {user_input}")  # Debug: User input
    response = chat_with_agent(user_input)  # Get agent's response
    print(f"[DEBUG] Agent response: {response}")  # Debug: Agent response
    st.session_state.chat_history.append(("🧑 Broker", user_input))  # Add user input to chat history
    st.session_state.chat_history.append(("🤖 Agent", response))     # Add agent response to chat history

# Display the chat history
for speaker, msg in st.session_state.chat_history:
    print(f"[DEBUG] Displaying message - {speaker}: {msg}")  # Debug: Display message
    st.markdown(f"**{speaker}:** {msg}")