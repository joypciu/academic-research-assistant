import hashlib
import json
from typing import Dict, Any, List
import streamlit as st
import datetime
import time

def generate_file_hash(file_content: bytes) -> str:
    """Generate hash for uploaded file to detect duplicates"""
    return hashlib.md5(file_content).hexdigest()

def format_timestamp(timestamp: float) -> str:
    """Format timestamp for display"""
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    return text[:max_length] + "..." if len(text) > max_length else text

def validate_pdf_file(uploaded_file) -> bool:
    """Validate if uploaded file is a proper PDF"""
    if uploaded_file.type != "application/pdf":
        st.error("Invalid file type. Please upload a PDF file.")
        return False
    
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File size too large. Maximum 10MB allowed.")
        return False
    
    # Check for valid PDF header
    try:
        uploaded_file.seek(0)
        header = uploaded_file.read(4)
        if not header.startswith(b'%PDF'):
            st.error("Invalid PDF file format.")
            return False
        uploaded_file.seek(0)  # Reset file pointer
        return True
    except Exception as e:
        st.error(f"Error validating PDF: {str(e)}")
        return False

def export_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """Export chat history as JSON string"""
    export_data = {
        "chat_history": [
            {
                "type": msg["type"],
                "content": msg["content"],
                "timestamp": format_timestamp(msg["timestamp"]),
                "sources": msg.get("sources", []),
                "retrieved_chunks": msg.get("retrieved_chunks", 0)
            } for msg in chat_history
        ],
        "export_timestamp": format_timestamp(time.time())
    }
    return json.dumps(export_data, indent=2)

#@st.cache_data
# def load_sample_questions() -> List[str]:
#     """Load sample research questions"""
#     return [
#         "What methodologies are used across these papers?",
#         "Compare the evaluation metrics used in these studies",
#         "What datasets were used for experiments?",
#         "What are the main contributions of each paper?",
#         "Identify research gaps mentioned in the papers",
#         "Compare performance results across different approaches",
#         "What are the common limitations discussed?",
#         "How do the related work sections compare?",
#         "What future work is suggested in these papers?",
#         "Which papers cite similar prior work?"
#     ]