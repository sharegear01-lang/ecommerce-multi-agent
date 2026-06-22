"""
Vector Store — manages the ChromaDB persistent vector store for the RAG system.

Handles initialization, ingestion of documents, and provides a configured
retriever for agent use.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from src.rag.loader import DataLoader

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages the ChromaDB vector store.

    On first run, loads sample data from JSON files, creates embeddings,
    and persists them to disk. On subsequent runs, loads the existing store.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL

        # Initialize embeddings
        logger.info("Loading embedding model: %s", self.embedding_model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # The Chroma collection instance
        self._collection: Optional[Chroma] = None

    @property
    def collection(self) -> Chroma:
        """Get or lazily initialize the Chroma collection."""
        if self._collection is None:
            self._collection = self._load_or_create()
        return self._collection

    def _load_or_create(self) -> Chroma:
        """
        Load existing ChromaDB or create a new one with ingested data.

        Returns:
            A Chroma vector store instance.
        """
        persist_path = Path(self.persist_dir)

        if persist_path.exists() and any(persist_path.iterdir()):
            logger.info("Loading existing ChromaDB from: %s", self.persist_dir)
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="ecommerce_kb",
            )

        logger.info("No existing ChromaDB found. Initializing from sample data...")
        return self._initialize_store()

    def _initialize_store(self) -> Chroma:
        """
        Create a new ChromaDB and ingest all sample data.

        Returns:
            A new Chroma vector store with ingested documents.
        """
        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Load documents from JSON
        loader = DataLoader()
        documents = loader.load_all()

        if not documents:
            logger.warning("No documents loaded. Creating empty store.")
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="ecommerce_kb",
            )

        # Ingest into Chroma
        logger.info("Ingesting %d documents into ChromaDB...", len(documents))
        collection = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="ecommerce_kb",
        )
        logger.info("ChromaDB initialized with %d documents", len(documents))
        return collection

    def reset(self) -> None:
        """Delete and reinitialize the vector store."""
        import shutil
        shutil.rmtree(self.persist_dir, ignore_errors=True)
        self._collection = None
        logger.info("Vector store reset. Will re-initialize on next access.")
        self._collection = self._load_or_create()
