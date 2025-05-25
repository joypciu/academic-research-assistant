import streamlit as st
import os
import tempfile
from typing import List
import logging
from dotenv import load_dotenv
import time
import pandas as pd
import json
import time

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

# Modern CSS styling
def load_css(theme: str = "Light"):
    css = """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #1a1a1a;
    }

    /* Theme-aware adjustments */
    """
    if theme == "Dark":
        css += """
        .stApp {
            color: #e5e7eb;
        }
        .main-container {
            background: rgba(31, 41, 55, 0.95);
        }
        .chat-container, .metric-container, .source-card {
            background: #374151;
            color: #e5e7eb;
        }
        .source-meta {
            color: #d1d5db;
        }
        .streamlit-expanderHeader {
            background: #4b5563;
            color: #e5e7eb;
        }
        """
    else:
        css += """
        .stApp {
            color: #1a1a1a;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.95);
        }
        .chat-container, .metric-container, .source-card {
            background: #ffffff;
        }
        .source-meta {
            color: #6b7280;
        }
        .streamlit-expanderHeader {
            background: #f8fafc;
            color: #1f2937;
        }
        """
    css += """
    /* Main container */
    .main-container {
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }

    .main-header p {
        color: #e5e7eb;
        font-size: 1.2rem;
        font-weight: 400;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
    }

    /* Upload area */
    .upload-container {
        border: 2px dashed #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: #f9fafb;
        transition: all 0.3s ease;
    }

    .upload-container:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }

    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #0ea5e9;
        color: #1f2937;
    }

    .success-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #22c55e;
        color: #1f2937;
    }

    .warning-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
        color: #1f2937;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
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
    }

    /* Metrics */
    .metric-container {
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Source citations */
    .source-card {
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #3b82f6;
    }

    .source-title {
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }

    /* Footer */
    .footer-container {
        text-align: center;
        color: #ffffff;
        padding: 2rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
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
        'theme': 'Light'
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

def display_year_chart(papers_overview):
    """Display bar chart of publication years"""
    years = [paper['year'] for paper in papers_overview.values() if paper['year'] != 'Unknown']
    if years:
        year_counts = pd.Series(years).value_counts().sort_index()
        st.markdown("### Publication Years Distribution")
        st.markdown(f"""
        ```chartjs
        {{
            "type": "bar",
            "data": {{
                "labels": {json.dumps(year_counts.index.tolist())},
                "datasets": [{{
                    "label": "Number of Papers",
                    "data": {json.dumps(year_counts.values.tolist())},
                    "backgroundColor": "#667eea",
                    "borderColor": "#3b82f6",
                    "borderWidth": 1
                }}]
            }},
            "options": {{
                "scales": {{
                    "y": {{
                        "beginAtZero": true,
                        "title": {{ "display": true, "text": "Number of Papers" }}
                    }},
                    "x": {{
                        "title": {{ "display": true, "text": "Publication Year" }}
                    }}
                }}
            }}
        }}
        ```
        """, unsafe_allow_html=True)

def display_paper_stats():
    """Display statistics about processed papers"""
    if not st.session_state.processed_papers:
        return
    
    papers_overview = st.session_state.rag_pipeline.get_papers_overview()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #3b82f6; margin: 0;">📄</h3>
            <h2 style="margin: 0;">{len(papers_overview)}</h2>
            <p style="color: #6b7280; margin: 0;">Papers Loaded</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_chunks = sum(paper['chunk_count'] for paper in papers_overview.values())
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #10b981; margin: 0;">🧩</h3>
            <h2 style="margin: 0;">{total_chunks}</h2>
            <p style="color: #6b7280; margin: 0;">Text Chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        all_sections = set()
        for paper in papers_overview.values():
            all_sections.update(paper['sections'])
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #f59e0b; margin: 0;">📑</h3>
            <h2 style="margin: 0;">{len(all_sections)}</h2>
            <p style="color: #6b7280; margin: 0;">Unique Sections</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        years = set(paper['year'] for paper in papers_overview.values() if paper['year'] != 'Unknown')
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #8b5cf6; margin: 0;">📅</h3>
            <h2 style="margin: 0;">{len(years) if years else 1}</h2>
            <p style="color: #6b7280; margin: 0;">Publication Years</p>
        </div>
        """, unsafe_allow_html=True)
    
    display_year_chart(papers_overview)

def display_papers_overview():
    """Display detailed overview of loaded papers"""
    if not st.session_state.processed_papers:
        return
    
    papers_overview = st.session_state.rag_pipeline.get_papers_overview()
    
    st.markdown("### 📚 Loaded Research Papers")
    
    for filename, paper_info in papers_overview.items():
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
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1rem; border-radius: 18px 18px 4px 18px; 
                        max-width: 70%; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <strong>You:</strong> {message['content']}<br>
                <span style="font-size: 0.8rem; opacity: 0.7;">{timestamp}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 1rem;">
            <div style="background: white; border: 1px solid #e5e7eb; 
                        padding: 1rem; border-radius: 18px 18px 18px 4px; 
                        max-width: 80%; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem; margin-right: 0.5rem;">🤖</span>
                    <strong style="color: #3b82f6;">Research Assistant</strong>
                    <span style="margin-left: auto; font-size: 0.8rem; color: #6b7280;">
                        {message['retrieved_chunks']} sources analyzed | {timestamp}
                    </span>
                </div>
                <div style="color: #374151;">{message['content']}</div>
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
                        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #4b5563;">
                            {source['content_preview']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Create RAG pipeline
    rag_pipeline = create_rag_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Controls")
        
        # Theme toggle
        st.markdown("#### 🎨 Theme")
        theme = st.selectbox("Select Theme", ["Light", "Dark"], key="theme_selector")
        load_css(theme)
        st.session_state.theme = theme
        
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
                    
                    documents = processor.process_pdf(uploaded_file, uploaded_file.name)
                    if documents:
                        all_documents.extend(documents)
                        st.session_state.processed_papers[uploaded_file.name] = len(documents)
                        processed_count += 1
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if all_documents:
                status_text.text("Adding to knowledge base...")
                rag_pipeline.add_papers(all_documents)
                
                st.success(f"✅ Processed {processed_count} new papers!")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("⚠️ No new papers to process.")
        
        # Clear button
        if st.session_state.processed_papers and st.button("🗑️ Clear All Papers", use_container_width=True, help="Clear all processed papers"):
            rag_pipeline.clear_knowledge_base()
            st.session_state.processed_papers = {}
            st.session_state.chat_history = []
            st.success("🧹 Knowledge base cleared!")
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
            # Placeholder for feedback storage (e.g., save to file or database)
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
                filter_section = search_sections[0] if len(search_sections) == 1 else None
                result = rag_pipeline.query(
                    user_question, 
                    k=num_sources,
                    filter_section=filter_section
                )
            
            st.session_state.chat_history.append({
                "type": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "retrieved_chunks": result["retrieved_chunks"],
                "timestamp": time.time()
            })
            
            st.rerun()
        
        if st.session_state.chat_history and st.button("🧹 Clear Chat History", help="Clear all chat messages"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-container">
        <p>Built with ❤️ using Streamlit, LangChain, FAISS, and Gemma API</p>
        <p>Perfect for researchers, students, and anyone working with academic literature</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()