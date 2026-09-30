from typing import Any

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from lectra.domain.retrieval import Chunk, RetrievedChunk


class QdrantRetriever:
    """Retrieves Lectra chunks using a Qdrant vector store."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: Embeddings,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = "lectra_benchmark",
    ) -> None:
        self._chunks = {chunk.id: chunk for chunk in chunks}
        self._embeddings = embeddings
        self._collection_name = collection_name

        self._client = QdrantClient(url=url)

        self._create_collection(chunks)
        self._index_chunks(chunks)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve the most similar chunks for a query."""
        query_vector = self._embeddings.embed_query(query)

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        retrieved_chunks = []

        for result in results:
            payload = result.payload or {}
            chunk_id = payload["chunk_id"]

            chunk = self._chunks[chunk_id]

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=result.score,
                )
            )

        return retrieved_chunks

    def _create_collection(self, chunks: list[Chunk]) -> None:
        """Create the Qdrant collection if it does not exist."""
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)

        vector_size = len(
            self._embeddings.embed_documents(
                [chunks[0].content],
            )[0]
        )

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _index_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and index all chunks in Qdrant."""
        vectors = self._embeddings.embed_documents(
            [chunk.content for chunk in chunks],
        )

        points = [
            models.PointStruct(
                id=index,
                vector=vector,
                payload=self._build_payload(chunk),
            )
            for index, (chunk, vector) in enumerate(
                zip(chunks, vectors, strict=True),
            )
        ]

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    @staticmethod
    def _build_payload(chunk: Chunk) -> dict[str, Any]:
        """Build the Qdrant payload for a chunk."""
        return {
            "chunk_id": chunk.id,
        }
