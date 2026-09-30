from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from lectra.domain.retrieval import Chunk, RetrievedChunk
from lectra.infrastructure.embeddings.ollama import OllamaEmbeddingProvider


class LangChainRetriever:
    """Retrieves Lectra chunks using a LangChain vector store."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._vector_store = self._build_vector_store(chunks)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve the most similar chunks for a query."""
        results = self._vector_store.similarity_search_with_score(
            query,
            k=top_k,
        )

        return [
            RetrievedChunk(
                chunk=document.metadata["chunk"],
                score=score,
            )
            for document, score in results
        ]

    @staticmethod
    def _build_vector_store(
        chunks: list[Chunk],
    ) -> InMemoryVectorStore:
        documents = [
            Document(
                page_content=chunk.content,
                metadata={"chunk": chunk},
            )
            for chunk in chunks
        ]

        vector_store = InMemoryVectorStore(OllamaEmbeddingProvider())
        vector_store.add_documents(documents)

        return vector_store
