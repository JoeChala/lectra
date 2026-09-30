import hashlib

from langchain_core.embeddings import Embeddings


class DeterministicEmbeddings(Embeddings):
    """Creates deterministic vectors for retrieval experiments."""

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create vectors for multiple documents."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Create a deterministic vector for one piece of text."""
        digest = hashlib.sha256(text.lower().encode()).digest()

        vector = []
        for index in range(self.dimensions):
            byte = digest[index % len(digest)]
            vector.append(byte / 255.0)

        return vector
