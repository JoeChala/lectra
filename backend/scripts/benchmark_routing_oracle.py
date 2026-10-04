from dataclasses import dataclass

from lectra.domain.retrieval_strategy import RetrievalStrategy
from lectra.evaluation.corpus import BENCHMARK_CHUNKS
from lectra.evaluation.retrieval_cases import ALL_RETRIEVAL_CASES
from lectra.infrastructure.embeddings.ollama import OllamaEmbeddingProvider
from lectra.infrastructure.retrieval.bm25 import BM25Retriever
from lectra.infrastructure.retrieval.hybrid import HybridRetriever
from lectra.infrastructure.routing.laya import LayaQueryRouter
from lectra.infrastructure.vector_store.langchain_memory import LangChainRetriever
from lectra.services.retriever import Retriever


@dataclass(frozen=True)
class StrategyResult:
    """Stores the retrieval outcome for one strategy."""

    strategy: RetrievalStrategy
    top_1_hit: bool
    first_relevant_rank: int | None


@dataclass
class QueryTypeStats:
    """Stores retrieval statistics for one query type."""

    total: int = 0
    dense_top_1_hits: int = 0
    bm25_top_1_hits: int = 0
    hybrid_top_1_hits: int = 0
    laya_top_1_hits: int = 0
    oracle_top_1_hits: int = 0
    oracle_solvable_cases: int = 0
    laya_achieved_oracle: int = 0


def evaluate_strategy(
    strategy: RetrievalStrategy,
    retriever: Retriever,
    query: str,
    relevant_chunk_ids: tuple[str, ...],
) -> StrategyResult:
    """Evaluate one retrieval strategy for a query."""

    results = retriever.retrieve(
        query,
        top_k=3,
    )

    ranked_ids = [result.chunk.id for result in results]

    relevant_ids = set(relevant_chunk_ids)

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

    return StrategyResult(
        strategy=strategy,
        top_1_hit=top_1_hit,
        first_relevant_rank=first_relevant_rank,
    )


def main() -> None:
    embeddings = OllamaEmbeddingProvider()

    dense_retriever = LangChainRetriever(
        BENCHMARK_CHUNKS,
        embeddings,
    )

    bm25_retriever = BM25Retriever(
        BENCHMARK_CHUNKS,
    )

    hybrid_retriever = HybridRetriever(
        retrievers=[
            dense_retriever,
            bm25_retriever,
        ],
    )

    retrievers = {
        RetrievalStrategy.DENSE: dense_retriever,
        RetrievalStrategy.BM25: bm25_retriever,
        RetrievalStrategy.HYBRID: hybrid_retriever,
    }

    router = LayaQueryRouter()

    laya_top_1_hits = 0
    oracle_top_1_hits = 0
    laya_achieved_oracle = 0
    oracle_solvable_cases = 0

    stats_by_type: dict[str, QueryTypeStats] = {}

    print("=== Laya Routing Oracle Benchmark ===")

    for case in ALL_RETRIEVAL_CASES:
        decision = router.route(case.query)

        strategy_results = [
            evaluate_strategy(
                strategy,
                retriever,
                case.query,
                case.relevant_chunk_ids,
            )
            for strategy, retriever in retrievers.items()
        ]

        oracle_top_1 = any(result.top_1_hit for result in strategy_results)

        laya_result = next(
            result
            for result in strategy_results
            if result.strategy is decision.strategy
        )

        laya_top_1_hits += laya_result.top_1_hit
        oracle_top_1_hits += oracle_top_1

        if oracle_top_1:
            oracle_solvable_cases += 1

        if oracle_top_1 and laya_result.top_1_hit:
            laya_achieved_oracle += 1

        stats = stats_by_type.setdefault(
            case.query_type,
            QueryTypeStats(),
        )

        stats.total += 1

        for result in strategy_results:
            if not result.top_1_hit:
                continue

            if result.strategy is RetrievalStrategy.DENSE:
                stats.dense_top_1_hits += 1
            elif result.strategy is RetrievalStrategy.BM25:
                stats.bm25_top_1_hits += 1
            elif result.strategy is RetrievalStrategy.HYBRID:
                stats.hybrid_top_1_hits += 1

        if laya_result.top_1_hit:
            stats.laya_top_1_hits += 1

        if oracle_top_1:
            stats.oracle_top_1_hits += 1
            stats.oracle_solvable_cases += 1

        if oracle_top_1 and laya_result.top_1_hit:
            stats.laya_achieved_oracle += 1

        print(f"\nQuery: {case.query}")
        print(f"Topic: {case.topic}")
        print(f"Type: {case.query_type}")
        print(f"Laya strategy: {decision.strategy}")
        print(f"Laya top-1: {laya_result.top_1_hit}")
        print(f"Laya first relevant rank: {laya_result.first_relevant_rank}")

        print("\nStrategies:")

        for result in strategy_results:
            print(
                f"  {result.strategy}: "
                f"top-1={result.top_1_hit}, "
                f"first-relevant-rank={result.first_relevant_rank}"
            )

        print(f"\nOracle top-1: {oracle_top_1}")
        print(f"Laya achieved oracle: {oracle_top_1 and laya_result.top_1_hit}")

    total_cases = len(ALL_RETRIEVAL_CASES)

    print("\n=== Oracle Summary ===")
    print(f"Laya Top-1: {laya_top_1_hits / total_cases:.2%}")
    print(f"Oracle Top-1: {oracle_top_1_hits / total_cases:.2%}")

    if oracle_solvable_cases:
        print(
            f"Laya achieved oracle: {laya_achieved_oracle / oracle_solvable_cases:.2%}"
        )

    print(f"Oracle-solvable cases: {oracle_solvable_cases}/{total_cases}")

    print("\n=== Results by Query Type ===")

    for query_type, stats in stats_by_type.items():
        print(f"\n{query_type} ({stats.total} cases)")

        print(f"  Dense: {stats.dense_top_1_hits / stats.total:.2%}")
        print(f"  BM25: {stats.bm25_top_1_hits / stats.total:.2%}")
        print(f"  Hybrid: {stats.hybrid_top_1_hits / stats.total:.2%}")
        print(f"  Laya: {stats.laya_top_1_hits / stats.total:.2%}")
        print(f"  Oracle: {stats.oracle_top_1_hits / stats.total:.2%}")

        if stats.oracle_solvable_cases:
            print(
                "  Laya achieved oracle: "
                f"{stats.laya_achieved_oracle / stats.oracle_solvable_cases:.2%}"
            )


if __name__ == "__main__":
    main()
