from lectra.domain.retrieval import Chunk, RetrievedChunk


def test_chunk_creation() -> None:
    """Verify a chunk keeps its document relationship and content."""
    chunk = Chunk(
        id="chunk-001",
        document_id="doc-001",
        content="A process is a program in execution.",
        metadata={"page": 12},
    )

    assert chunk.id == "chunk-001"
    assert chunk.document_id == "doc-001"
    assert chunk.metadata["page"] == 12


def test_retrieved_chunk_contains_score() -> None:
    """Verify retrieval results preserve the similarity score."""
    chunk = Chunk(
        id="chunk-001",
        document_id="doc-001",
        content="A process is a program in execution.",
        metadata={"page": 12},
    )

    result = RetrievedChunk(chunk=chunk, score=0.91)

    assert result.chunk == chunk
    assert result.score == 0.91
