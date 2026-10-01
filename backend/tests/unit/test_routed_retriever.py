from unittest.mock import Mock

from lectra.domain.retrieval import Chunk, RetrievedChunk
from lectra.domain.retrieval_strategy import RetrievalStrategy
from lectra.services.routed_retriever import RoutedRetriever


def test_routed_retriever_uses_selected_strategy():
    router = Mock()
    dense_retriever = Mock()
    bm25_retriever = Mock()

    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="test",
        metadata={},
    )

    expected = [
        RetrievedChunk(
            chunk=chunk,
            score=1.0,
        )
    ]

    router.route.return_value.strategy = RetrievalStrategy.BM25
    bm25_retriever.retrieve.return_value = expected

    retriever = RoutedRetriever(
        router=router,
        retrievers={
            RetrievalStrategy.DENSE: dense_retriever,
            RetrievalStrategy.BM25: bm25_retriever,
        },
    )

    result = retriever.retrieve("test query", top_k=3)

    assert result == expected
    bm25_retriever.retrieve.assert_called_once_with(
        "test query",
        top_k=3,
    )
    dense_retriever.retrieve.assert_not_called()
