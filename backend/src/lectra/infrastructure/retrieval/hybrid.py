from collections import defaultdict

from lectra.domain.retrieval import RetrievedChunk
from lectra.services.retriever import Retriever


class HybridRetriever:
    """Combines multiple retrievers using Reciprocal Rank Fusion."""

    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:
        if not retrievers:
            raise ValueError("At least one retriever is required.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        self._retrievers = retrievers
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve and fuse results from all configured retrievers."""

        fused_scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, RetrievedChunk] = {}

        for retriever in self._retrievers:
            results = retriever.retrieve(
                query,
                top_k=self._candidate_k,
            )

            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.id

                fused_scores[chunk_id] += 1 / (self._rrf_k + rank)

                chunks[chunk_id] = result

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )[:top_k]

        return [
            RetrievedChunk(
                chunk=chunks[chunk_id].chunk,
                score=fused_scores[chunk_id],
            )
            for chunk_id in ranked_ids
        ]
