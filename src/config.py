"""
Configuration settings for Live Web RAG system.
Uses NVIDIA API for embeddings and LLM.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SeleniumConfig:
    """Selenium WebDriver configuration."""
    headless: bool = False
    window_size: tuple = (1920, 1080)
    page_load_timeout: int = 30
    implicit_wait: int = 10


@dataclass
class EmbeddingConfig:
    """NVIDIA Embedding model configuration."""
    model: str = "nvidia/nv-embedqa-e5-v5"
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class LLMConfig:
    """NVIDIA LLM configuration."""
    model: str = "meta/llama-3.1-8b-instruct"
    temperature: float = 0.3
    max_tokens: int = 1000


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    provider: str = "chromadb"
    collection_name: str = "web_rag"
    persist_directory: str = "./chroma_db"


@dataclass
class CacheConfig:
    """Caching configuration."""
    enabled: bool = True
    ttl_seconds: float = 300  # 5 minutes


@dataclass
class WatcherConfig:
    """DOM watcher configuration."""
    poll_interval: float = 1.0
    change_debounce: float = 0.5


@dataclass
class Config:
    """Main configuration container."""
    # NVIDIA API key (set via NVIDIA_API_KEY env var)
    nvidia_api_key: Optional[str] = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY"))
    
    # Sub-configurations
    selenium: SeleniumConfig = field(default_factory=SeleniumConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("NVIDIA_API_KEY"):
            config.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            
        if os.getenv("LLM_MODEL"):
            config.llm.model = os.getenv("LLM_MODEL")
            
        if os.getenv("EMBEDDING_MODEL"):
            config.embedding.model = os.getenv("EMBEDDING_MODEL")
            
        if os.getenv("HEADLESS"):
            config.selenium.headless = os.getenv("HEADLESS", "").lower() == "true"
            
        if os.getenv("CHROMA_PERSIST_DIR"):
            config.vector_db.persist_directory = os.getenv("CHROMA_PERSIST_DIR")
            
        return config


# Default configuration instance
default_config = Config()
