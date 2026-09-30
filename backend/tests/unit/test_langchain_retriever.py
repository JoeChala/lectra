from lectra.domain.retrieval import Chunk
from lectra.infrastructure.langchain.retriever import LangChainRetriever


def test_retriever_returns_requested_number_of_results() -> None:
    """Verify the retriever returns at most the requested number of chunks."""
    chunks = [
        Chunk(
            id="chunk-1",
            document_id="os",
            content="A process is a program in execution.",
            metadata={"page": 10},
        ),
        Chunk(
            id="chunk-2",
            document_id="os",
            content="Deadlock occurs when processes wait indefinitely.",
            metadata={"page": 20},
        ),
        Chunk(
            id="chunk-3",
            document_id="os",
            content="CPU scheduling determines which process runs next.",
            metadata={"page": 30},
        ),
    ]

    retriever = LangChainRetriever(chunks)

    results = retriever.retrieve("process scheduling", top_k=2)

    assert len(results) == 2
    assert all(result.chunk in chunks for result in results)


def test_retriever_preserves_chunk_metadata() -> None:
    """Verify retrieval returns the original Lectra chunk."""
    chunk = Chunk(
        id="chunk-1",
        document_id="os",
        content="A process is a program in execution.",
        metadata={"page": 10},
    )

    retriever = LangChainRetriever([chunk])

    results = retriever.retrieve("process", top_k=1)

    assert results[0].chunk.id == "chunk-1"
    assert results[0].chunk.document_id == "os"
    assert results[0].chunk.metadata["page"] == 10
