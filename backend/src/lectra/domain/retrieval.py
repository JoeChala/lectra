from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """Represents a retrievable piece of a document."""

    id: str
    document_id: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a chunk returned by a retrieval operation."""

    chunk: Chunk
    score: float
