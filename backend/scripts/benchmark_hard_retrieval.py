from langchain_core.embeddings import Embeddings

from lectra.evaluation.corpus import BENCHMARK_CHUNKS
from lectra.evaluation.diagnostics import print_retrieval_diagnostics
from lectra.evaluation.metrics import evaluate
from lectra.evaluation.retrieval_cases import HARD_RETRIEVAL_CASES
from lectra.infrastructure.embeddings.ollama import OllamaEmbeddingProvider
from lectra.infrastructure.retrieval.bm25 import BM25Retriever
from lectra.infrastructure.vector_store.langchain_memory import LangChainRetriever


def build_dense_retriever(
    embeddings: Embeddings,
) -> LangChainRetriever:
    """Build the dense retriever used for the benchmark."""

    return LangChainRetriever(
        BENCHMARK_CHUNKS,
        embeddings=embeddings,
    )


def build_bm25_retriever() -> BM25Retriever:
    """Build the BM25 retriever used for the benchmark."""

    return BM25Retriever(BENCHMARK_CHUNKS)


def main() -> None:
    """Run the hard retrieval benchmark."""

    dense_retriever = build_dense_retriever(
        OllamaEmbeddingProvider(),
    )

    bm25_retriever = build_bm25_retriever()

    print_retrieval_diagnostics(
        "Nomic",
        dense_retriever,
        HARD_RETRIEVAL_CASES,
    )

    print_retrieval_diagnostics(
        "BM25",
        bm25_retriever,
        HARD_RETRIEVAL_CASES,
    )

    dense_result = evaluate(
        "nomic-embed-text",
        dense_retriever,
        HARD_RETRIEVAL_CASES,
    )

    bm25_result = evaluate(
        "bm25",
        bm25_retriever,
        HARD_RETRIEVAL_CASES,
    )

    print("\n=== Hard Retrieval Summary ===")

    for result in (dense_result, bm25_result):
        print(
            f"{result.name}: "
            f"Top-1={result.top_1_accuracy:.2%}, "
            f"Recall@3={result.recall_at_3:.2%}, "
            f"MRR={result.mean_reciprocal_rank:.3f}"
        )


if __name__ == "__main__":
    main()
