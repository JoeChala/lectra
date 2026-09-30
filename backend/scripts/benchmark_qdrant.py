from langchain_core.embeddings import Embeddings

from lectra.evaluation.corpus import BENCHMARK_CHUNKS
from lectra.evaluation.retrieval_cases import RETRIEVAL_CASES
from lectra.infrastructure.embeddings.ollama import OllamaEmbeddingProvider
from lectra.infrastructure.vector_store.qdrant import QdrantRetriever


def main() -> None:
    """Run the Qdrant retrieval benchmark."""
    embeddings: Embeddings = OllamaEmbeddingProvider()

    retriever = QdrantRetriever(
        BENCHMARK_CHUNKS,
        embeddings,
        collection_name="lectra_benchmark",
    )

    top_1_hits = 0
    recall_at_3_sum = 0.0
    reciprocal_rank_sum = 0.0

    for case in RETRIEVAL_CASES:
        results = retriever.retrieve(
            case.query,
            top_k=3,
        )

        ranked_ids = [result.chunk.id for result in results]

        relevant_ids = set(case.relevant_chunk_ids)

        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(
                ranked_ids,
                start=1,
            )
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
        print(f"Relevant: {case.relevant_chunk_ids}")
        print(f"Retrieved: {ranked_ids}")
        print(f"First relevant rank: {first_relevant_rank}")

    total = len(RETRIEVAL_CASES)

    print("\n=== Qdrant Summary ===")
    print(f"Top-1: {top_1_hits / total:.2%}")
    print(f"Recall@3: {recall_at_3_sum / total:.2%}")
    print(f"MRR: {reciprocal_rank_sum / total:.3f}")


if __name__ == "__main__":
    main()
