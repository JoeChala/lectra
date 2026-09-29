"""Application-specific exceptions used throughout Study RAG."""


class LectraRAGError(Exception):
    """Base exception for all application-specific errors."""


class ConfigurationError(LectraRAGError):
    """Raised when application configuration is invalid."""


class DocumentProcessingError(LectraRAGError):
    """Raised when a document cannot be parsed or processed."""


class EmbeddingError(LectraRAGError):
    """Raised when embedding generation fails."""


class VectorStoreError(LectraRAGError):
    """Raised when a vector-store operation fails."""


class RetrievalError(LectraRAGError):
    """Raised when document retrieval fails."""


class LLMError(LectraRAGError):
    """Raised when an LLM operation fails."""


class VisionModelError(LectraRAGError):
    """Raised when a vision-model operation fails."""


class CitationError(LectraRAGError):
    """Raised when citation creation or validation fails."""
