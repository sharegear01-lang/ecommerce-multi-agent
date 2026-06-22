"""
Centralized configuration for the E-commerce Multi-Agent System.
Reads from .env file and OS environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    # DeepSeek API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))

    # RAG Settings
    MAX_RETRIEVAL_DOCS: int = int(os.getenv("MAX_RETRIEVAL_DOCS", "3"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Application
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # LangSmith Tracing (optional)
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "ecommerce-mas")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    # Data files (relative to project root)
    DATA_DIR: Path = PROJECT_ROOT / "src" / "rag" / "data"

    @classmethod
    def setup_langsmith(cls) -> None:
        """
        Enable LangSmith tracing by setting OS environment variables.
        Must be called BEFORE any LangChain imports to take effect.
        Safe to call even when tracing is disabled (no-op).
        """
        if cls.LANGCHAIN_TRACING_V2.lower() == "true" and cls.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = cls.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = cls.LANGCHAIN_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = cls.LANGCHAIN_ENDPOINT
            print(f"LangSmith tracing enabled — project: {cls.LANGCHAIN_PROJECT}")
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            print("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")

    @classmethod
    def validate(cls) -> list[str]:
        """Validate required settings. Returns list of missing configurations."""
        errors = []
        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY is not set. Please configure it in .env")
        return errors

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (masking sensitive values)."""
        print(f"LLM Provider: DeepSeek")
        print(f"Model: {cls.DEEPSEEK_MODEL}")
        print(f"Base URL: {cls.DEEPSEEK_BASE_URL}")
        print(f"Embedding: {cls.EMBEDDING_MODEL}")
        print(f"ChromaDB: {cls.CHROMA_PERSIST_DIR}")
        print(f"Server: {cls.HOST}:{cls.PORT}")
        print(f"LangSmith: {'enabled' if cls.LANGCHAIN_TRACING_V2.lower() == 'true' else 'disabled'}")


# Module-level instance for easy import
settings = Settings()
