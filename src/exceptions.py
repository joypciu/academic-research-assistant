"""Custom exceptions for the research assistant"""

class ResearchAssistantError(Exception):
    """Base exception for research assistant"""
    pass

class DocumentProcessingError(ResearchAssistantError):
    """Raised when document processing fails"""
    pass

class VectorStoreError(ResearchAssistantError):
    """Raised when vector store operations fail"""
    pass

class RAGPipelineError(ResearchAssistantError):
    """Raised when RAG pipeline operations fail"""
    pass

class APIError(ResearchAssistantError):
    """Raised when API calls fail"""
    pass