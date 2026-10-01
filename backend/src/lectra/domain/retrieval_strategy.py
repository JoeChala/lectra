from enum import StrEnum


class RetrievalStrategy(StrEnum):
    """Defines the retrieval strategies available to Lectra."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
