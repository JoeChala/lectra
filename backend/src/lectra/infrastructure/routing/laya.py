from typing import TypedDict, cast

from laya import Router

from lectra.domain.retrieval_strategy import RetrievalStrategy
from lectra.domain.routing import RetrievalDecision
from lectra.services.router import QueryRouter


class LayaChoiceAnswer(TypedDict):
    """Represents a Laya choice answer."""

    type: str
    choice: str
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float


class LayaPrediction(TypedDict):
    """Represents the portion of a Laya prediction used by Lectra."""

    answers: dict[str, LayaChoiceAnswer]


class LayaQueryRouter(QueryRouter):
    """Routes Lectra queries using a local Laya decision model."""

    _QUESTION_ID = "retrieval_strategy"

    _QUESTIONS = {
        _QUESTION_ID: {
            "type": "choice",
            "instructions": (
                "Which retrieval strategy is most appropriate for answering this query?"
            ),
            "criteria": {
                "dense": (
                    "Use semantic vector retrieval when the query is "
                    "conceptual, paraphrased, or depends on semantic meaning."
                ),
                "bm25": (
                    "Use lexical retrieval when exact terminology, names, "
                    "identifiers, algorithms, or specific technical terms "
                    "are important."
                ),
                "hybrid": (
                    "Use both semantic and lexical retrieval when the query "
                    "benefits from both semantic similarity and exact "
                    "terminology matching."
                ),
            },
        }
    }

    def __init__(self) -> None:
        self._router = Router()

    def route(self, query: str) -> RetrievalDecision:
        """Select a retrieval strategy for a query."""

        if not query.strip():
            raise ValueError("Query must not be empty.")

        result = cast(
            LayaPrediction,
            self._router.predict(
                query,
                self._QUESTIONS,
                model="typed-decisions",
            ),
        )

        answer = result["answers"][self._QUESTION_ID]

        strategy = RetrievalStrategy(answer["choice"])

        return RetrievalDecision(
            strategy=strategy,
        )
