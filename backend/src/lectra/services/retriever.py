from typing import Protocol

from lectra.domain.retrieval import RetrievedChunk


class Retriever(Protocol):
    """Defines the retrieval capability required by Lectra."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query."""
        ...
