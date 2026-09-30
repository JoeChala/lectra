from lectra.domain.retrieval import Chunk
from lectra.infrastructure.embeddings.ollama import OllamaEmbeddingProvider
from lectra.infrastructure.vector_store.langchain_memory import LangChainRetriever


def main() -> None:
    chunks = [
        Chunk(
            id="chunk-1",
            document_id="os",
            content="A process is a program in execution.",
            metadata={"page": 10, "topic": "processes"},
        ),
        Chunk(
            id="chunk-2",
            document_id="os",
            content="Deadlock occurs when processes wait indefinitely.",
            metadata={"page": 20, "topic": "deadlocks"},
        ),
        Chunk(
            id="chunk-3",
            document_id="os",
            content="CPU scheduling determines which process runs next.",
            metadata={"page": 30, "topic": "scheduling"},
        ),
    ]

    embeddings = OllamaEmbeddingProvider()

    for chunk, vector in zip(
        chunks,
        embeddings.embed_documents([chunk.content for chunk in chunks]),
        strict=False,
    ):
        print(f"{chunk.id}:")
        print(f"  dimensions: {len(vector)}")
        print(f"  first 5 values: {vector[:5]}")
        print()

    retriever = LangChainRetriever(chunks, embeddings)

    query = "How does CPU scheduling work?"
    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}")
    print("\nResults:")

    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. {result.chunk.id}")
        print(f"   Score: {result.score}")
        print(f"   Content: {result.chunk.content}")
        print(f"   Metadata: {result.chunk.metadata}")


if __name__ == "__main__":
    main()
