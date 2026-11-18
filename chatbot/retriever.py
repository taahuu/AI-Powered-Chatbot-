"""Retriever interface and implementations for RAG and external knowledge bases.

To plug in a real vector database (e.g. Chroma, FAISS, Pinecone) or external knowledge base:
1. Subclass BaseRetriever:
   class ChromaRetriever(BaseRetriever):
       def __init__(self, collection):
           self.collection = collection

       def retrieve(self, query: str, k: int = 3) -> list[Document]:
           results = self.collection.query(query_texts=[query], n_results=k)
           return [
               Document(content=doc, source=meta.get("source", ""))
               for doc, meta in zip(results["documents"][0], results["metadatas"][0])
           ]
2. Pass the custom retriever instance to ChatService:
   service = ChatService(..., retriever=ChromaRetriever(collection))
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a retrieved document or context passage.

    Attributes:
        content: The text content of the document.
        source: Optional source or citation identifier.
    """

    content: str
    source: str = ""


class BaseRetriever(ABC):
    """Abstract base class for all document retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """Retrieve top-k relevant documents for the given query.

        Args:
            query: The search query string.
            k: Maximum number of documents to return.

        Returns:
            List of Document objects matching the query.
        """
        pass


class NullRetriever(BaseRetriever):
    """Default no-op retriever that returns an empty list.

    Leaves prompt formatting unchanged and adds zero overhead when RAG is not configured.
    """

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """Return an empty list of documents."""
        return []
