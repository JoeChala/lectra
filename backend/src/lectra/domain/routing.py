from dataclasses import dataclass

from lectra.domain.retrieval_strategy import RetrievalStrategy


@dataclass(frozen=True)
class RetrievalDecision:
    """Represents a retrieval strategy selected for a query."""

    strategy: RetrievalStrategy
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float
