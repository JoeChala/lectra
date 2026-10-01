import pytest

from lectra.domain.retrieval import Chunk, RetrievedChunk
from lectra.infrastructure.retrieval.hybrid import HybridRetriever


class FakeRetriever:
    """Provides predefined retrieval results for testing."""

    def __init__(
        self,
        results: list[RetrievedChunk],
    ) -> None:
        self._results = results

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return predefined retrieval results."""

        return self._results[:top_k]


def make_chunk(chunk_id: str) -> Chunk:
    """Create a test chunk."""

    return Chunk(
        id=chunk_id,
        document_id="test",
        content=chunk_id,
        metadata={},
    )


def make_result(
    chunk_id: str,
    score: float,
) -> RetrievedChunk:
    """Create a test retrieval result."""

    return RetrievedChunk(
        chunk=make_chunk(chunk_id),
        score=score,
    )


def test_hybrid_retriever_combines_results() -> None:
    retriever_a = FakeRetriever(
        [
            make_result("chunk-a", 0.9),
            make_result("chunk-b", 0.8),
        ]
    )

    retriever_b = FakeRetriever(
        [
            make_result("chunk-b", 0.9),
            make_result("chunk-c", 0.8),
        ]
    )

    hybrid = HybridRetriever(
        [retriever_a, retriever_b],
    )

    results = hybrid.retrieve(
        "test query",
        top_k=3,
    )

    assert [result.chunk.id for result in results] == [
        "chunk-b",
        "chunk-a",
        "chunk-c",
    ]


def test_hybrid_retriever_respects_top_k() -> None:
    retriever = FakeRetriever(
        [
            make_result("chunk-a", 0.9),
            make_result("chunk-b", 0.8),
            make_result("chunk-c", 0.7),
        ]
    )

    hybrid = HybridRetriever([retriever])

    results = hybrid.retrieve(
        "test query",
        top_k=2,
    )

    assert len(results) == 2


def test_hybrid_retriever_requires_retrievers() -> None:
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_hybrid_retriever_requires_positive_rrf_k() -> None:
    retriever = FakeRetriever(
        [make_result("chunk-a", 1.0)],
    )

    with pytest.raises(ValueError):
        HybridRetriever(
            [retriever],
            rrf_k=0,
        )
