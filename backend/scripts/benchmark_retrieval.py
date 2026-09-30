from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from lectra.evaluation.corpus import BENCHMARK_CHUNKS
from lectra.evaluation.retrieval_cases import RETRIEVAL_CASES
from lectra.infrastructure.embeddings.deterministic import DeterministicEmbeddings
from lectra.infrastructure.embeddings.ollama import OllamaEmbeddingProvider
from lectra.infrastructure.vector_store.langchain_memory import LangChainRetriever


@dataclass(frozen=True)
class BenchmarkResult:
    """Contains aggregate retrieval benchmark results."""

    model_name: str
    top_1_accuracy: float
    recall_at_3: float
    mean_reciprocal_rank: float


def evaluate(
    model_name: str,
    embeddings: Embeddings,
) -> BenchmarkResult:
    """Evaluate one embedding provider against the retrieval cases."""
    retriever = LangChainRetriever(
        BENCHMARK_CHUNKS,
        embeddings=embeddings,
    )

    top_1_hits = 0
    recall_at_3_sum = 0.0
    reciprocal_rank_sum = 0.0

    print(f"\n=== {model_name} ===")

    for case in RETRIEVAL_CASES:
        results = retriever.retrieve(case.query, top_k=3)
        ranked_ids = [result.chunk.id for result in results]

        relevant_ids = set(case.relevant_chunk_ids)

        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(ranked_ids, start=1)
            if chunk_id in relevant_ids
        ]

        first_relevant_rank = relevant_ranks[0] if relevant_ranks else None

        if ranked_ids and ranked_ids[0] in relevant_ids:
            top_1_hits += 1

        relevant_retrieved = sum(chunk_id in relevant_ids for chunk_id in ranked_ids)

        recall_at_3_sum += relevant_retrieved / len(relevant_ids)

        if first_relevant_rank is not None:
            reciprocal_rank_sum += 1 / first_relevant_rank

        print(f"\nQuery: {case.query}")
        print(f"Type: {case.query_type}")
        print(f"Relevant: {case.relevant_chunk_ids}")
        print(f"Retrieved: {ranked_ids}")
        print(f"First relevant rank: {first_relevant_rank}")

    total = len(RETRIEVAL_CASES)

    return BenchmarkResult(
        model_name=model_name,
        top_1_accuracy=top_1_hits / total,
        recall_at_3=recall_at_3_sum / total,
        mean_reciprocal_rank=reciprocal_rank_sum / total,
    )


def main() -> None:
    """Run the retrieval benchmark."""
    results = [
        evaluate(
            "deterministic",
            DeterministicEmbeddings(),
        ),
        evaluate(
            "nomic-embed-text",
            OllamaEmbeddingProvider(),
        ),
    ]

    print("\n=== Summary ===")

    for result in results:
        print(
            f"{result.model_name}: "
            f"Top-1={result.top_1_accuracy:.2%}, "
            f"Recall@3={result.recall_at_3:.2%}, "
            f"MRR={result.mean_reciprocal_rank:.3f}"
        )


if __name__ == "__main__":
    main()
