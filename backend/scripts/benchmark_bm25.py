from lectra.evaluation.corpus import BENCHMARK_CHUNKS
from lectra.evaluation.metrics import evaluate
from lectra.evaluation.retrieval_cases import RETRIEVAL_CASES
from lectra.infrastructure.retrieval.bm25 import BM25Retriever


def main() -> None:
    """Run the BM25 retrieval benchmark."""

    retriever = BM25Retriever(BENCHMARK_CHUNKS)

    result = evaluate(
        "bm25",
        retriever,
        RETRIEVAL_CASES,
    )

    print("\n=== BM25 Summary ===")
    print(
        f"{result.name}: "
        f"Top-1={result.top_1_accuracy:.2%}, "
        f"Recall@3={result.recall_at_3:.2%}, "
        f"MRR={result.mean_reciprocal_rank:.3f}"
    )


if __name__ == "__main__":
    main()
