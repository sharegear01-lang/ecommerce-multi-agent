"""
Retriever — domain-specific retrieval interface for agents.

Provides type-filtered retrieval methods so each agent can query
only the relevant subset of the knowledge base.
"""

import logging
from typing import Optional

from langchain_chroma import Chroma

from config import settings

logger = logging.getLogger(__name__)


class Retriever:
    """
    Domain-specific retrieval wrapper around ChromaDB.

    Each method filters by document type metadata to return
    only relevant results for the querying agent.
    """

    def __init__(self, vector_store: Chroma, max_docs: Optional[int] = None):
        """
        Args:
            vector_store: Initialized Chroma vector store instance.
            max_docs: Maximum number of documents to retrieve per query.
        """
        self.store = vector_store
        self.max_docs = max_docs or settings.MAX_RETRIEVAL_DOCS

    def _build_filter(self, base_filters: dict) -> Optional[dict]:
        """
        Build a ChromaDB-compatible where filter.

        Single condition: {"key": "value"}
        Multiple conditions: {"$and": [{"key1": "value1"}, {"key2": "value2"}]}
        """
        conditions = [{"type": v} if k == "type" else {k: v} for k, v in base_filters.items()]
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def retrieve_products(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> list:
        """
        Retrieve product documents relevant to the query.

        Args:
            query: Natural language query about products.
            top_k: Number of results (defaults to settings.MAX_RETRIEVAL_DOCS).
            category: Optional category filter.
            brand: Optional brand filter.

        Returns:
            List of matching Document objects.
        """
        k = top_k or self.max_docs
        filter_base = {"type": "product"}
        if category:
            filter_base["category"] = category
        if brand:
            filter_base["brand"] = brand

        chroma_filter = self._build_filter(filter_base)
        logger.debug("Retrieving products: query='%s', k=%d, filter=%s", query, k, chroma_filter)
        results = self.store.similarity_search(query, k=k, filter=chroma_filter)
        return results

    def retrieve_policies(
        self,
        query: str = "",
        policy_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list:
        """
        Retrieve policy documents.

        Args:
            query: Natural language query about policies.
            policy_type: Optional specific policy type
                         (return_policy, exchange_policy, shipping_policy, etc.).
            top_k: Number of results.

        Returns:
            List of matching Document objects.
        """
        k = top_k or self.max_docs
        filter_base = {"type": "policy"}
        if policy_type:
            filter_base["policy_type"] = policy_type

        chroma_filter = self._build_filter(filter_base)
        # If no query, do a broader search
        search_query = query or "退换货 配送 保修 支付"
        logger.debug("Retrieving policies: query='%s', filter=%s", search_query, chroma_filter)
        results = self.store.similarity_search(search_query, k=k, filter=chroma_filter)
        return results

    def retrieve_faq(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list:
        """
        Retrieve FAQ documents relevant to the query.

        Args:
            query: Natural language query matching FAQ topics.
            top_k: Number of results.

        Returns:
            List of matching Document objects.
        """
        k = top_k or max(2, self.max_docs - 1)
        chroma_filter = self._build_filter({"type": "faq"})
        logger.debug("Retrieving FAQ: query='%s', k=%d", query, k)
        results = self.store.similarity_search(query, k=k, filter=chroma_filter)
        return results

    def retrieve_all(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list:
        """
        Retrieve documents across all types (general search).

        Args:
            query: Natural language query.
            top_k: Number of results.

        Returns:
            List of matching Document objects.
        """
        k = top_k or self.max_docs
        logger.debug("Retrieving all: query='%s', k=%d", query, k)
        results = self.store.similarity_search(query, k=k)
        return results

    def format_context(self, docs: list) -> str:
        """
        Format retrieved documents into a single context string for LLM prompts.

        Args:
            docs: List of retrieved Document objects.

        Returns:
            Formatted string with document contents separated by dividers.
        """
        if not docs:
            return "暂无相关参考信息。"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("type", "unknown")
            parts.append(f"--- 参考资料 {i} (类型: {source}) ---\n{doc.page_content}")
        return "\n\n".join(parts)
