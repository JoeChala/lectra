from dataclasses import dataclass

from lectra.evaluation.retrieval_cases import RetrievalCase
from lectra.services.retriever import Retriever


@dataclass(frozen=True)
class BenchmarkResult:
    """Contains aggregate retrieval benchmark results."""

    name: str
    top_1_accuracy: float
    recall_at_3: float
    mean_reciprocal_rank: float


def evaluate(
    name: str,
    retriever: Retriever,
    cases: list[RetrievalCase],
) -> BenchmarkResult:
    """Evaluate a retriever against retrieval cases."""

    top_1_hits = 0
    recall_at_3_sum = 0.0
    reciprocal_rank_sum = 0.0

    for case in cases:
        results = retriever.retrieve(
            case.query,
            top_k=3,
        )

        ranked_ids = [result.chunk.id for result in results]

        relevant_ids = set(case.relevant_chunk_ids)

        if ranked_ids and ranked_ids[0] in relevant_ids:
            top_1_hits += 1

        relevant_retrieved = sum(chunk_id in relevant_ids for chunk_id in ranked_ids)

        recall_at_3_sum += relevant_retrieved / len(relevant_ids)

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

        if first_relevant_rank is not None:
            reciprocal_rank_sum += 1 / first_relevant_rank

    total = len(cases)

    return BenchmarkResult(
        name=name,
        top_1_accuracy=top_1_hits / total,
        recall_at_3=recall_at_3_sum / total,
        mean_reciprocal_rank=reciprocal_rank_sum / total,
    )
