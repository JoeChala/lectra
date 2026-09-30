from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


class OllamaEmbeddingProvider(Embeddings):
    """Provides embeddings using a local Ollama model."""

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._embeddings = OllamaEmbeddings(
            model=model,
            base_url=base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for multiple documents."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Create an embedding for a query."""
        return self._embeddings.embed_query(text)
