import os
import streamlit as st
from dotenv import load_dotenv
from src.agent_response import chat_with_agent
from src.data_loader import load_chat_data, load_policy_docs
from src.embedding_index import build_or_load_index

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Safe Streamlit secrets access (avoid exception if no secrets file locally)
if not os.getenv("OPENAI_API_KEY"):
    try:
        secret_key = None
        # st.secrets may raise if no secrets config; guard with try
        try:
            secret_key = st.secrets.get("OPENAI_API_KEY")  # returns None if missing
        except Exception:
            secret_key = None
        if secret_key:
            os.environ["OPENAI_API_KEY"] = secret_key
    except Exception:
        pass

api_key_present = bool(os.getenv("OPENAI_API_KEY"))
if not api_key_present:
    st.warning("OPENAI_API_KEY not set. Add it as a Streamlit secret or environment variable.")

# Set Streamlit app title with enhanced meta information
st.set_page_config(
    page_title="Agenverse - Underwriting AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Agenverse is an AI-powered underwriting assistant that helps insurance professionals with policy analysis and coverage questions."
    }
)

# Add Open Graph meta tags for better social media sharing
st.markdown("""
<meta property="og:title" content="Agenverse - AI Underwriting Assistant">
<meta property="og:description" content="AI-powered underwriting assistant for insurance professionals. Get instant policy analysis and coverage insights.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://agenverse.streamlit.app">
<meta name="description" content="Agenverse is an AI-powered underwriting assistant that helps insurance professionals with policy analysis and coverage questions.">
<meta name="keywords" content="AI, underwriting, insurance, policy analysis, artificial intelligence">
""", unsafe_allow_html=True)

# Beautiful header with custom styling
st.markdown("""
<div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">
    <div style="
        background: linear-gradient(90deg, #1f4e79, #2e7bcf, #4a90e2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        cursor: default;
        user-select: none;
    ">
        🛡️ Agenverse
    </div>
    <div style="
        color: #5a6c7d;
        font-size: 1.4rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        letter-spacing: 0.5px;
        cursor: default;
        user-select: none;
    ">
        Your Intelligent Underwriting Assistant
    </div>
    <div style="
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #2e7bcf, #4a90e2);
        margin: 1rem auto;
        border-radius: 2px;
    "></div>
    <p style="
        color: #7a8a99;
        font-size: 1rem;
        margin: 0;
        font-style: italic;
    ">
        Ask underwriting questions • AI-powered broker support • Policy & chat insights
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_index():
    chat_data = load_chat_data("data/chat_data/chat_data.xlsx")
    policy_docs = load_policy_docs("data/policy_documents")
    return build_or_load_index(chat_data, policy_docs)

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    print("[DEBUG] Initialized chat_history in session state")  # Debug: Session state init

# Input box for user to ask a question
user_input = st.text_input("Ask a question about underwriting, coverage, or policy:")

# Sidebar with enhanced sections
st.sidebar.markdown("""
<style>
.developer-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
    text-align: center;
    color: white;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
}
.about-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
    color: white;
    box-shadow: 0 4px 16px rgba(240, 147, 251, 0.2);
}
.tech-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
    color: white;
    box-shadow: 0 4px 16px rgba(79, 172, 254, 0.2);
}
.connect-card {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
    color: white;
    box-shadow: 0 4px 16px rgba(67, 233, 123, 0.2);
}
.connect-card a {
    color: white !important;
    text-decoration: none;
    font-weight: 600;
}
.connect-card a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# About & Tech section (expandable)
with st.sidebar.expander("🛡️ About Agenverse", expanded=False):
    # About section
    st.markdown("""
    <div class="about-card">
        <h3 style="margin-top: 0;">🛡️ About Agenverse</h3>
        <p style="font-size: 0.9rem; line-height: 1.4; margin: 0;">
            Agenverse is an AI-powered underwriting co-pilot for insurance businesses, intelligently handling broker queries, quotes, renewals, and amendments while leveraging historical chat data and underwriting rules to optimize workflows and drive business growth.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack section
    st.markdown("""
    <div class="tech-card">
        <h3 style="margin-top: 0;">🔧 Tech Stack</h3>
        <p style="font-size: 0.85rem; line-height: 1.3; margin: 0;">
            • Python<br>
            • Streamlit<br>
            • OpenAI GPT<br>
            • LangChain<br>
            • FAISS<br>
            • Vector Embeddings<br>
            • PyMuPDF<br>
            • OCR Processing<br>
            • RAG (Retrieval-Augmented Generation)
        </p>
    </div>
    """, unsafe_allow_html=True)

# Policy Documents section
with st.sidebar.expander("📋 Policy Documents", expanded=False):
    import os
    
    policy_base_path = "data/policy_documents"
    
    # Function to get user-friendly document names
    def get_friendly_name(filename, category):
        name_mapping = {
            # Commercial Combined
            "PolicyWording.pdf": "Policy Wording Document",
            "SummaryofCover.pdf": "Summary of Cover",
            "zurich_sme_commercial_combined.pdf": "Commercial Combined Policy Guide",
            
            # Property owners
            "721395.pdf": "Property Owners Policy - Standard",
            "721396.pdf": "Property Owners Policy - Enhanced", 
            "zurich_sme_property_owners.pdf": "Property Owners Policy Guide",
            
            # Shop
            "721688.pdf": "Shop Insurance Policy - Basic",
            "721733.pdf": "Shop Insurance Policy - Comprehensive",
            "zurich_sme_shop.pdf": "Shop Insurance Policy Guide",
            
            # Small Fleet
            "Fleet Handbook.pdf": "Fleet Management Handbook",
            "Policy Summary.pdf": "Fleet Policy Summary",
            "Policy Wording.pdf": "Fleet Policy Wording",
            "zurich_sme_small_fleet (1).pdf": "Small Fleet Policy Guide",
            
            # Trades and professions
            "721689.pdf": "Trades & Professions Policy - Standard",
            "721748.pdf": "Trades & Professions Policy - Premium",
            "zurich_sme_trades_professions.pdf": "Trades & Professions Policy Guide"
        }
        
        return name_mapping.get(filename, filename.replace('.pdf', '').replace('_', ' ').title())
    
    if os.path.exists(policy_base_path):
        st.markdown("**Available Policy Documents:**")
        
        # Get all policy categories
        for category in sorted(os.listdir(policy_base_path)):
            category_path = os.path.join(policy_base_path, category)
            
            # Skip hidden files and non-directories
            if category.startswith('.') or not os.path.isdir(category_path):
                continue
                
            st.markdown(f"**{category}:**")
            
            # List all PDFs in this category
            for file in sorted(os.listdir(category_path)):
                if file.endswith('.pdf'):
                    file_path = os.path.join(category_path, file)
                    friendly_name = get_friendly_name(file, category)
                    
                    # Create download button for each PDF
                    try:
                        with open(file_path, "rb") as pdf_file:
                            pdf_data = pdf_file.read()
                            st.download_button(
                                label=f"📄 {friendly_name}",
                                data=pdf_data,
                                file_name=file,
                                mime="application/pdf",
                                key=f"download_{category}_{file}"  # Unique key for each button
                            )
                    except Exception as e:
                        st.text(f"❌ {file} (unavailable)")
            
            st.markdown("---")  # Separator between categories
    else:
        st.info("No policy documents found.")

# Developer section (expandable) - moved to last
with st.sidebar.expander("👨‍💻 Meet the Developer", expanded=False):
    # Display developer image first
    try:
        st.markdown("""
        <div style="text-align: center; margin: 10px 0;">
            <img src="data:image/png;base64,{}" style="width: 160px; height: auto; border-radius: 50%; border: 1px solid #a8b8d8;">
        </div>
        """.format(
            __import__('base64').b64encode(open("images/developer-image.png", "rb").read()).decode()
        ), unsafe_allow_html=True)
    except:
        st.info("Developer image not found")
    
    st.markdown("""
    <div class="developer-card">
        <h3 style="margin-top: 0; font-size: 1.4rem;">Rajul Shakywar</h3>
        <p style="font-size: 1.1rem; margin: 5px 0; opacity: 0.9;">Software Engineer (GenAI)</p>
        <p style="font-size: 0.9rem; margin: 15px 0 0 0; opacity: 0.8;">Passionate about building AI solutions that transform business processes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Connect section inside developer expandable
    st.markdown("""
    <div class="connect-card">
        <h3 style="margin-top: 0;">🔗 Connect</h3>
        <p style="margin: 10px 0;">
            <a href="https://www.linkedin.com/in/rajul-shakywar/" target="_blank">💼 LinkedIn</a><br>
            <a href="https://github.com/rajulshakya777/AgentVerse" target="_blank">🐱 GitHub</a><br>
            <a href="https://rajulshakya777.github.io/portfolio/" target="_blank">🌐 Portfolio</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if user_input:
    st.session_state.chat_history.append(("👤 You", user_input))
    with st.spinner("Thinking..."):
        try:
            _ = get_index()  # ensure index cached
            # Pass full history except the last user message just added (history includes it already)
            response = chat_with_agent(user_input, history=st.session_state.chat_history[:-1])
        except Exception as e:
            import traceback, textwrap
            tb = traceback.format_exc()
            msg = f"Error: {e}"
            st.error(msg)
            response = msg
        finally:
            st.session_state.chat_history.append(("⚡️ Agenverse", response))

# Display the chat history
with st.container():
    for speaker, msg in reversed(st.session_state.chat_history):
        formatted = msg.replace("\n", "  \n")
        st.markdown(f"**{speaker}:**  \n{formatted}")