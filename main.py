import streamlit as st
import os
import tempfile
from typing import List
import logging
from dotenv import load_dotenv
import time
import pandas as pd
import plotly.express as px

# Import custom modules
from src.document_processor import AcademicPaperProcessor
from src.rag_pipeline import ResearchRAGPipeline
from src.utils import export_chat_history, format_timestamp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, eye-friendly design
def load_css():
    css = """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* CSS Variables for consistent theming */
    :root {
        --background: #FFFFFF;
        --text-color: #1A1A1A;
        --container-bg: #F9FAFB;
        --card-bg: #FFFFFF;
        --meta-color: #6B7280;
        --expander-bg: #F8FAFC;
        --expander-text: #2F3437;
        --button-bg: #667EEA;
        --button-text: #FFFFFF;
        --accent-color: #667EEA;
    }

    /* Main app styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--background);
        color: var(--text-color);
    }

    /* Target the specific container for suggested questions */
    .element-container:has(.suggested-questions) {
        background: var(--card-bg) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* Alternative approach - target by data-testid if available */
    [data-testid="stVerticalBlock"] > div:has(.suggested-questions) {
        background: var(--card-bg) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* Style the markdown container that holds suggested questions */
    .suggested-questions-container {
        background: var(--card-bg) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* Style buttons within suggested questions */
    .suggested-questions-container .stButton > button,
    .element-container:has(.suggested-questions) .stButton > button {
        background: #EFF6FF !important;
        color: #1F2937 !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .suggested-questions-container .stButton > button:hover,
    .element-container:has(.suggested-questions) .stButton > button:hover {
        background: #DBEAFE !important;
        transform: translateY(-2px) !important;
    }

    /* Main container */
    .main-container {
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        background: var(--container-bg);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        color: var(--accent-color);
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: var(--meta-color);
        font-size: 1.2rem;
        font-weight: 400;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: var(--container-bg);
        border-radius: 15px;
        padding: 1rem;
    }

    /* Upload area */
    .upload-container {
        border: 2px dashed #E5E7EB;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: var(--card-bg);
        transition: all 0.3s ease;
    }

    .upload-container:hover {
        border-color: var(--accent-color);
        background: #EFF6FF;
    }

    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--accent-color);
        color: var(--text-color);
    }

    .success-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #22C55E;
        color: var(--text-color);
    }

    .warning-card {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #F59E0B;
        color: var(--text-color);
    }

    /* Buttons */
    .stButton > button {
        background: var(--button-bg);
        color: var(--button-text);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -1px rgba(0, 0, 0, 0.2);
    }

    /* Chat styling */
    .chat-container {
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        background: var(--card-bg);
        color: var(--text-color);
    }

    /* Metrics */
    .metric-container {
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        background: var(--card-bg);
        color: var(--text-color);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: var(--button-bg);
    }

    /* Source citations */
    .source-card {
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid var(--accent-color);
        color: var(--text-color);
        background: var(--card-bg);
    }

    .source-title {
        font-weight: 600;
        color: var(--accent-color);
        margin-bottom: 0.5rem;
    }

    .source-meta {
        color: var(--meta-color);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--expander-bg);
        color: var(--expander-text);
    }

    /* Footer */
    .footer-container {
        text-align: center;
        color: var(--meta-color);
        padding: 2rem;
    }

    /* Clear chat button styling */
    button[data-testid="baseButton-secondary"]:has-text("🧹 Clear Chat History"),
    div[data-testid="stButton"] button:has-text("🧹 Clear Chat History") {
        background: #F87171 !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    button[data-testid="baseButton-secondary"]:has-text("🧹 Clear Chat History"):hover,
    div[data-testid="stButton"] button:has-text("🧹 Clear Chat History"):hover {
        background: #EF4444 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
    }

    /* Alternative approach using key-based targeting */
    div[data-testid="stButton"] button[aria-label*="Clear Chat History"] {
        background: #F87171 !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    div[data-testid="stButton"] button[aria-label*="Clear Chat History"]:hover {
        background: #EF4444 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'rag_pipeline': None,
        'processed_papers': {},
        'chat_history': [],
        'processing_status': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_rag_pipeline():
    """Initialize RAG pipeline with API key"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("🔑 GEMINI_API_KEY not found in environment variables.")
            st.stop()
        
        if st.session_state.rag_pipeline is None:
            st.session_state.rag_pipeline = ResearchRAGPipeline(api_key)
        
        return st.session_state.rag_pipeline
    except Exception as e:
        st.error(f"❌ Error initializing RAG pipeline: {str(e)}")
        st.stop()

def display_year_chart(papers_summary):
    """Display bar chart of publication years using Plotly"""
    years = [paper['year'] for paper in papers_summary.values() if paper['year'] != 'Unknown']
    if years:
        year_counts = pd.Series(years).value_counts().sort_index()
        df = pd.DataFrame({'Year': year_counts.index, 'Number of Papers': year_counts.values})
        fig = px.bar(
            df,
            x='Year',
            y='Number of Papers',
            title="Publication Years Distribution",
            color_discrete_sequence=['#667EEA'],
            template='plotly_white'
        )
        fig.update_layout(
            xaxis_title="Publication Year",
            yaxis_title="Number of Papers",
            yaxis=dict(tickmode='linear'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

def display_paper_stats():
    """Display statistics about processed papers"""
    if not st.session_state.processed_papers:
        return
    
    papers_summary = st.session_state.rag_pipeline.get_papers_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-container">
                <h3 style="color: var(--accent-color); margin: 0;">Papers</h3>
                <h2 style="margin: 0;">{len(papers_summary)}</h2>
                <p style="color: var(--meta-color); margin: 0;">Papers Loaded</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        total_chunks = sum(paper['chunk_count'] for paper in papers_summary.values())
        st.markdown(
            f"""
            <div class="metric-container">
                <h3 style="color: #10b981; margin: 0;">Chunks</h3>
                <h2 style="margin: 0;">{total_chunks}</h2>
                <p style="color: var(--meta-color); margin: 0;">Text Chunks</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        all_sections = set()
        for paper in papers_summary.values():
            all_sections.update(paper['sections'])
        st.markdown(
            f"""
            <div class="metric-container">
                <h3 style="color: #f59e0b; margin: 0;">Sections</h3>
                <h2 style="margin: 0;">{len(all_sections)}</h2>
                <p style="color: var(--meta-color); margin: 0;">Unique Sections</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        years = set(paper['year'] for paper in papers_summary.values() if paper['year'] != 'Unknown')
        st.markdown(
            f"""
            <div class="metric-container">
                <h3 style="color: #8b5cf6; margin: 0;">Years</h3>
                <h2 style="margin: 0;">{len(years) if years else 1}</h2>
                <p style="color: var(--meta-color); margin: 0;">Publication Years</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    display_year_chart(papers_summary)

def display_papers_overview():
    """Display detailed overview of loaded papers"""
    if not st.session_state.processed_papers:
        return
    
    papers_summary = st.session_state.rag_pipeline.get_papers_summary()
    
    st.markdown("### 📚 Loaded Research Papers")
    
    for filename, paper_info in papers_summary.items():
        with st.expander(f"📄 {paper_info['title'][:80]}...", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Authors:** {paper_info['authors']}")
                st.markdown(f"**Year:** {paper_info['year']}")
                st.markdown(f"**Filename:** {filename}")
                st.markdown(f"**Text Chunks:** {paper_info['chunk_count']}")
            
            with col2:
                st.markdown("**Sections Found:**")
                for section in paper_info['sections']:
                    st.markdown(f"• {section}")

def display_suggested_questions():
    """Display suggested research questions"""
    if not st.session_state.processed_papers:
        return
    
    st.markdown("### 💡 Suggested Research Questions")
    
    suggestions = st.session_state.rag_pipeline.suggest_research_questions()
    
    if not suggestions:
        st.markdown("No relevant questions generated. Upload more papers to get suggestions.")
        return
    
    # Add custom CSS specifically for this section
    st.markdown("""
    <style>
    /* Override button styling for suggested questions */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background-color: #EFF6FF !important;
        color: #1F2937 !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
    }
    
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        background-color: #DBEAFE !important;
        color: #1F2937 !important;
        border: 1px solid #BFDBFE !important;
    }
    
    /* Target all buttons in this section */
    div[data-testid="column"] > div > div > button {
        background-color: #EFF6FF !important;
        color: #1F2937 !important;
        border: 1px solid #E5E7EB !important;
    }
    
    div[data-testid="column"] > div > div > button:hover {
        background-color: #DBEAFE !important;
        color: #1F2937 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, question in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(f"🤔 {question}", key=f"suggestion_{i}", use_container_width=True, help="Click to ask this question"):
                st.session_state.chat_history.append({
                    "type": "user",
                    "content": question,
                    "timestamp": time.time()
                })
                
                with st.spinner("🔍 Analyzing papers..."):
                    result = st.session_state.rag_pipeline.query(question)
                
                st.session_state.chat_history.append({
                    "type": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "retrieved_chunks": result["retrieved_chunks"],
                    "timestamp": time.time()
                })
                
                st.rerun()

def display_chat_message(message):
    """Display a chat message with proper styling and timestamp"""
    timestamp = format_timestamp(message["timestamp"])
    if message["type"] == "user":
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
            <div style="background: var(--button-bg); 
                        color: var(--button-text); padding: 1rem; border-radius: 18px 18px 4px 18px; 
                        max-width: 70%; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <strong>You:</strong> {message['content']}<br>
                <span style="font-size: 0.8rem; opacity: 0.7;">{timestamp}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 1rem;">
            <div style="background: var(--card-bg); border: 1px solid #E5E7EB; 
                        padding: 1rem; border-radius: 18px 18px 18px 4px; 
                        max-width: 80%; box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                        color: var(--text-color);">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem; margin-right: 0.5rem;">🤖</span>
                    <strong style="color: var(--accent-color);">Research Assistant</strong>
                    <span style="margin-left: auto; font-size: 0.8rem; color: var(--meta-color);">
                        {message['retrieved_chunks']} sources analyzed | {timestamp}
                    </span>
                </div>
                <div>{message['content']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if message.get('sources'):
            with st.expander(f"📚 View {len(message['sources'])} Sources", expanded=False):
                for source in message['sources']:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">📄 {source['title'][:100]}...</div>
                        <div class="source-meta">
                            <strong>Section:</strong> {source['section']} | 
                            <strong>Authors:</strong> {source['authors']} | 
                            <strong>Year:</strong> {source['year']} | 
                            <strong>Relevance:</strong> {source['relevance_score']}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: var(--meta-color);">
                            {source['content_preview']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load custom CSS
    load_css()
    
    # Create RAG pipeline
    rag_pipeline = create_rag_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Controls")
        
        # File uploader
        st.markdown("#### 📤 Upload Research Papers")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload academic papers in PDF format (max 10MB each)"
        )
        
        # Processing button
        if uploaded_files and st.button("🚀 Process Papers", use_container_width=True, help="Process uploaded PDFs"):
            processor = AcademicPaperProcessor()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_documents = []
            processed_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file.name not in st.session_state.processed_papers:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    logger.info(f"Processing file: {uploaded_file.name}")
                    
                    documents = processor.process_pdf(uploaded_file, uploaded_file.name)
                    if documents:
                        all_documents.extend(documents)
                        st.session_state.processed_papers[uploaded_file.name] = len(documents)
                        processed_count += 1
                        logger.info(f"Processed {uploaded_file.name}: {len(documents)} chunks")
                    else:
                        logger.warning(f"No documents extracted from {uploaded_file.name}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if all_documents:
                status_text.text("Adding to knowledge base...")
                rag_pipeline.add_papers(all_documents)
                logger.info(f"Added {len(all_documents)} documents to vector store")
                st.success(f"✅ Processed {processed_count} new papers!")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("⚠️ No new papers to process.")
                logger.warning("No documents added to vector store")
        
        # Clear button
        if st.session_state.processed_papers and st.button("🗑️ Clear All Papers", use_container_width=True, help="Clear all processed papers"):
            rag_pipeline.clear_knowledge_base()
            st.session_state.processed_papers = {}
            st.session_state.chat_history = []
            st.success("🧹 Knowledge base cleared!")
            logger.info("Cleared knowledge base")
            time.sleep(1)
            st.rerun()
        
        # Export analysis
        if st.session_state.chat_history:
            st.markdown("#### 📥 Export Analysis")
            chat_json = export_chat_history(st.session_state.chat_history)
            st.download_button(
                label="Download Chat History",
                data=chat_json,
                file_name=f"research_analysis_{int(time.time())}.json",
                mime="application/json",
                help="Download chat history as JSON"
            )
        
        # Query options
        st.markdown("#### ⚙️ Query Options")
        search_sections = st.multiselect(
            "Filter by sections",
            ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion"],
            help="Leave empty to search all sections"
        )
        
        num_sources = st.slider(
            "Number of sources to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="More sources provide richer context but slower processing"
        )
        
        # Feedback form
        st.markdown("#### 📢 Feedback")
        feedback = st.text_area("Share your feedback", key="feedback_input", help="Let us know how we can improve!")
        if st.button("Submit Feedback", help="Submit your feedback"):
            with open("feedback.txt", "a") as f:
                f.write(f"[{format_timestamp(time.time())}] {feedback}\n")
            st.success("Thank you for your feedback!")
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Academic Research Assistant</h1>
        <p>Upload research papers and ask intelligent questions powered by RAG + Gemma</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.session_state.processed_papers:
            display_paper_stats()
            display_papers_overview()
        else:
            st.markdown("""
            <div class="info-card">
                <h3>🚀 Getting Started</h3>
                <p>1. Upload research papers (PDF format) using the sidebar</p>
                <p>2. Click "Process Papers" to build your knowledge base</p>
                <p>3. Ask questions about methodologies, results, comparisons, and more!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.processed_papers:
            display_suggested_questions()
    
    # Chat interface
    if st.session_state.processed_papers:
        st.markdown("---")
        st.markdown("### 💬 Research Chat")
        
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                display_chat_message(message)
        
        user_question = st.chat_input(
            placeholder="Ask about methodologies, compare results, identify gaps, etc...",
            key="research_chat"
        )
        
        if user_question:
            st.session_state.chat_history.append({
                "type": "user",
                "content": user_question,
                "timestamp": time.time()
            })
            
            with st.spinner("🔍 Analyzing papers and generating response..."):
                filter_section = None
                result = rag_pipeline.query(
                    user_question, 
                    k=num_sources,
                    filter_section=filter_section
                )
                logger.info(f"Query result: {result['retrieved_chunks']} chunks retrieved for '{user_question}'")
            
            st.session_state.chat_history.append({
                "type": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "retrieved_chunks": result["retrieved_chunks"],
                "timestamp": time.time()
            })
            
            st.rerun()
        
   # Clear chat history button with proper styling
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:  # Center the button
            if st.button("🧹 Clear Chat History", help="Clear all chat messages", key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.rerun()

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-container">
        <p>Built with ❤️ using Streamlit, LangChain, FAISS, and Gemma API</p>
        <p>Made by Usman Gani Joy 👨‍💻. Orcid Link: https://orcid.org/0009-0003-9498-3828</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()