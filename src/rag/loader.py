"""
Data Loader — converts JSON sample data files into LangChain Document objects
suitable for ingestion into the ChromaDB vector store.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads JSON data files and converts them to LangChain Documents.

    Each record is split into appropriately-sized chunks with metadata
    to enable filtered retrieval by document type.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.data_dir = data_dir or settings.DATA_DIR
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "，", "；", " ", ""],
        )

    def load_all(self) -> list[Document]:
        """
        Load all data files and return a combined list of Documents.

        Returns:
            List of LangChain Document objects ready for vector store ingestion.
        """
        documents = []
        documents.extend(self.load_products())
        documents.extend(self.load_policies())
        documents.extend(self.load_faq())
        logger.info(
            "Loaded %d documents (%d products, %d policies, %d faq)",
            len(documents),
            len([d for d in documents if d.metadata.get("type") == "product"]),
            len([d for d in documents if d.metadata.get("type") == "policy"]),
            len([d for d in documents if d.metadata.get("type") == "faq"]),
        )
        return documents

    def load_products(self) -> list[Document]:
        """Load and convert product data into Documents."""
        filepath = self.data_dir / "products.json"
        if not filepath.exists():
            logger.warning("Products file not found: %s", filepath)
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            products = json.load(f)

        documents = []
        for product in products:
            pid = product["id"]

            # 1) Full product card
            specs_str = ", ".join(
                f"{k}: {v}" for k, v in product["specs"].items()
            )
            card_text = (
                f"商品名称: {product['name']}\n"
                f"品牌: {product['brand']}\n"
                f"类别: {product['category']}\n"
                f"售价: ¥{product['price']}\n"
                f"原价: ¥{product['original_price']}\n"
                f"库存: {product['inventory']}件\n"
                f"规格: {specs_str}\n"
                f"描述: {product['description']}\n"
                f"促销: {product.get('promotion', '暂无')}\n"
                f"标签: {', '.join(product.get('tags', []))}"
            )
            doc = Document(
                page_content=card_text,
                metadata={
                    "type": "product",
                    "product_id": pid,
                    "category": product["category"],
                    "brand": product["brand"],
                    "doc_kind": "card",
                },
            )
            # Split if too long
            documents.extend(self._split_doc(doc))

            # 2) Specs-only document (for spec-specific queries)
            specs_doc = Document(
                page_content=(
                    f"{product['name']} 规格参数:\n" + specs_str
                ),
                metadata={
                    "type": "product",
                    "product_id": pid,
                    "category": product["category"],
                    "brand": product["brand"],
                    "doc_kind": "specs",
                },
            )
            documents.append(specs_doc)

            # 3) Promotion document
            if product.get("promotion"):
                promo_doc = Document(
                    page_content=(
                        f"{product['name']} 促销信息: {product['promotion']}"
                    ),
                    metadata={
                        "type": "product",
                        "product_id": pid,
                        "category": product["category"],
                        "brand": product["brand"],
                        "doc_kind": "promotion",
                    },
                )
                documents.append(promo_doc)

        logger.info("Loaded %d products → %d documents", len(products), len(documents))
        return documents

    def load_policies(self) -> list[Document]:
        """Load and convert policy data into Documents."""
        filepath = self.data_dir / "policies.json"
        if not filepath.exists():
            logger.warning("Policies file not found: %s", filepath)
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            policies = json.load(f)

        documents = []
        for policy_key, policy_data in policies.items():
            # Flatten the policy dict into readable text
            lines = [f"【{policy_data.get('title', policy_key)}】"]
            for key, value in policy_data.items():
                if key == "title":
                    continue
                if isinstance(value, list):
                    lines.append(f"{key}: " + "; ".join(str(v) for v in value))
                elif isinstance(value, dict):
                    lines.append(f"{key}: " + "; ".join(
                        f"{k}: {v}" for k, v in value.items()
                    ))
                else:
                    lines.append(f"{key}: {value}")

            doc = Document(
                page_content="\n".join(lines),
                metadata={
                    "type": "policy",
                    "policy_type": policy_key,
                },
            )
            documents.extend(self._split_doc(doc))

        logger.info("Loaded %d policies → %d documents", len(policies), len(documents))
        return documents

    def load_faq(self) -> list[Document]:
        """Load and convert FAQ data into Documents."""
        filepath = self.data_dir / "faq.json"
        if not filepath.exists():
            logger.warning("FAQ file not found: %s", filepath)
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            faqs = json.load(f)

        documents = []
        for faq in faqs:
            doc = Document(
                page_content=f"问题: {faq['question']}\n回答: {faq['answer']}",
                metadata={
                    "type": "faq",
                    "faq_id": faq["id"],
                },
            )
            documents.append(doc)

        logger.info("Loaded %d FAQs → %d documents", len(faqs), len(documents))
        return documents

    def _split_doc(self, doc: Document) -> list[Document]:
        """Split a document if it exceeds chunk size, preserving metadata."""
        chunks = self.text_splitter.split_documents([doc])
        for chunk in chunks:
            # Inherit metadata from parent
            for key, value in doc.metadata.items():
                if key not in chunk.metadata:
                    chunk.metadata[key] = value
        return chunks if chunks else [doc]
