from lectra.infrastructure.routing.laya import LayaQueryRouter

QUERIES = [
    "What is a process in an operating system?",
    "What is the FIFO page replacement algorithm?",
    "Explain how virtual memory works.",
    "Compare FIFO and LRU page replacement algorithms.",
    "What happens during a context switch?",
    "What is the smallest unit of CPU execution inside a process?",
    "How can concurrent threads safely access shared resources?",
    "What is the difference between paging and virtual memory?",
]


def main() -> None:
    router = LayaQueryRouter()

    print("=== Laya Retrieval Routing ===")

    for query in QUERIES:
        decision = router.route(query)

        print(f"\nQuery: {query}")
        print(f"Strategy: {decision.strategy}")
        print(f"Probabilities: {decision.probabilities}")
        print(f"Confidence: {decision.confidence:.4f}")
        print(f"Answer confidence: {decision.answer_confidence:.4f}")


if __name__ == "__main__":
    main()
