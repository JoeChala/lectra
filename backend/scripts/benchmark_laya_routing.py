from lectra.domain.retrieval_strategy import RetrievalStrategy
from lectra.evaluation.corpus import BENCHMARK_CHUNKS
from lectra.evaluation.diagnostics import print_retrieval_diagnostics
from lectra.evaluation.metrics import evaluate
from lectra.evaluation.retrieval_cases import HARD_RETRIEVAL_CASES
from lectra.infrastructure.embeddings.ollama import OllamaEmbeddingProvider
from lectra.infrastructure.retrieval.bm25 import BM25Retriever
from lectra.infrastructure.retrieval.hybrid import HybridRetriever
from lectra.infrastructure.routing.laya import LayaQueryRouter
from lectra.infrastructure.vector_store.langchain_memory import LangChainRetriever
from lectra.services.routed_retriever import RoutedRetriever


def main() -> None:
    embeddings = OllamaEmbeddingProvider()

    dense_retriever = LangChainRetriever(
        BENCHMARK_CHUNKS,
        embeddings,
    )

    bm25_retriever = BM25Retriever(BENCHMARK_CHUNKS)

    hybrid_retriever = HybridRetriever(
        retrievers=[
            dense_retriever,
            bm25_retriever,
        ],
    )

    router = LayaQueryRouter()

    routed_retriever = RoutedRetriever(
        router=router,
        retrievers={
            RetrievalStrategy.DENSE: dense_retriever,
            RetrievalStrategy.BM25: bm25_retriever,
            RetrievalStrategy.HYBRID: hybrid_retriever,
        },
    )

    result = evaluate(
        name="laya-routed",
        retriever=routed_retriever,
        cases=HARD_RETRIEVAL_CASES,
    )

    print("\n=== Laya Routing Summary ===")
    print(f"Top-1: {result.top_1_accuracy:.2%}")
    print(f"Recall@3: {result.recall_at_3:.2%}")
    print(f"MRR: {result.mean_reciprocal_rank:.3f}")

    print_retrieval_diagnostics(
        name="Laya-Routed",
        retriever=routed_retriever,
        cases=HARD_RETRIEVAL_CASES,
    )

    print("\n=== Laya Strategy Decisions ===")

    for case in HARD_RETRIEVAL_CASES:
        decision = router.route(case.query)

        print(f"\nQuery: {case.query}")
        print(f"Strategy: {decision.strategy}")


if __name__ == "__main__":
    main()
