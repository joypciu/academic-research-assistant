"""
Academic Research Paper Assistant with RAG
A sophisticated tool for analyzing research papers using LangChain, FAISS, and Gemma API
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .document_processor import AcademicPaperProcessor
from .vector_store import FAISSVectorStore
from .rag_pipeline import ResearchRAGPipeline

__all__ = [
    "AcademicPaperProcessor",
    "FAISSVectorStore", 
    "ResearchRAGPipeline"
]