from typing import Protocol


class EmbeddingProvider(Protocol):
    """Defines the embedding capability required by Lectra."""

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Create embeddings for multiple documents."""
        ...

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Create an embedding for a query."""
        ...
