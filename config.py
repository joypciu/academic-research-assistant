import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AppConfig:
    """Application configuration"""
    
    # API Configuration
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    gemini_model: str = "gemma-3-12b-it"
    
    # Text Processing
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Generation Parameters
    max_output_tokens: int = 800
    temperature: float = 0.3
    top_p: float = 0.8
    
    # UI Configuration
    max_file_size_mb: int = 10
    max_files_per_upload: int = 10
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "app.log")
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.gemini_api_key:
            return False
        return True

# Global config instance
config = AppConfig()