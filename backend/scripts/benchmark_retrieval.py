from langchain_core.embeddings import Embeddings

from lectra.evaluation.corpus import BENCHMARK_CHUNKS
from lectra.evaluation.metrics import BenchmarkResult, evaluate
from lectra.evaluation.retrieval_cases import RETRIEVAL_CASES
from lectra.infrastructure.embeddings.deterministic import DeterministicEmbeddings
from lectra.infrastructure.embeddings.ollama import OllamaEmbeddingProvider
from lectra.infrastructure.vector_store.langchain_memory import LangChainRetriever


def run_benchmark(
    name: str,
    embeddings: Embeddings,
) -> BenchmarkResult:
    """Run the benchmark for one embedding provider."""

    retriever = LangChainRetriever(
        BENCHMARK_CHUNKS,
        embeddings=embeddings,
    )

    print(f"\n=== {name} ===")

    for case in RETRIEVAL_CASES:
        results = retriever.retrieve(
            case.query,
            top_k=3,
        )

        ranked_ids = [result.chunk.id for result in results]

        relevant_ids = set(case.relevant_chunk_ids)

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(
                    ranked_ids,
                    start=1,
                )
                if chunk_id in relevant_ids
            ),
            None,
        )

        print(f"\nQuery: {case.query}")
        print(f"Type: {case.query_type}")
        print(f"Relevant: {case.relevant_chunk_ids}")
        print(f"Retrieved: {ranked_ids}")
        print(f"First relevant rank: {first_relevant_rank}")

    return evaluate(
        name,
        retriever,
        RETRIEVAL_CASES,
    )


def main() -> None:
    """Run the retrieval benchmark."""

    results = [
        run_benchmark(
            "deterministic",
            DeterministicEmbeddings(),
        ),
        run_benchmark(
            "nomic-embed-text",
            OllamaEmbeddingProvider(),
        ),
    ]

    print("\n=== Summary ===")

    for result in results:
        print(
            f"{result.name}: "
            f"Top-1={result.top_1_accuracy:.2%}, "
            f"Recall@3={result.recall_at_3:.2%}, "
            f"MRR={result.mean_reciprocal_rank:.3f}"
        )


if __name__ == "__main__":
    main()
