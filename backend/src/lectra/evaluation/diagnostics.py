from lectra.evaluation.retrieval_cases import RetrievalCase
from lectra.services.retriever import Retriever


def print_retrieval_diagnostics(
    name: str,
    retriever: Retriever,
    cases: list[RetrievalCase],
) -> None:
    """Print per-query retrieval results for diagnostic analysis."""

    print(f"\n=== {name} ===")

    for case in cases:
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

        top_1_hit = bool(ranked_ids) and ranked_ids[0] in relevant_ids

        print(f"\nQuery: {case.query}")
        print(f"Type: {case.query_type}")
        print(f"Topic: {case.topic}")
        print(f"Relevant: {case.relevant_chunk_ids}")
        print(f"Retrieved: {ranked_ids}")
        print(f"Top-1 hit: {top_1_hit}")
        print(f"First relevant rank: {first_relevant_rank}")
