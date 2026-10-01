from lectra.evaluation.corpus import BENCHMARK_CHUNKS
from lectra.infrastructure.retrieval.bm25 import BM25Retriever


def test_bm25_retrieves_cpu_scheduling() -> None:
    retriever = BM25Retriever(BENCHMARK_CHUNKS)

    results = retriever.retrieve(
        "How does the operating system decide which process gets CPU time?",
        top_k=3,
    )

    assert results
    assert results[0].chunk.id == "os-scheduling"


def test_bm25_retrieves_page_replacement() -> None:
    retriever = BM25Retriever(BENCHMARK_CHUNKS)

    results = retriever.retrieve(
        "Which memory page should be removed when there is no free frame?",
        top_k=3,
    )

    assert results
    assert results[0].chunk.id == "os-page-replacement"


def test_bm25_respects_top_k() -> None:
    retriever = BM25Retriever(BENCHMARK_CHUNKS)

    results = retriever.retrieve(
        "operating system",
        top_k=3,
    )

    assert len(results) == 3
