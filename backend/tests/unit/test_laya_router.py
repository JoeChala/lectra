from unittest.mock import patch

import pytest

from lectra.domain.retrieval_strategy import RetrievalStrategy
from lectra.infrastructure.routing.laya import LayaQueryRouter


def test_laya_router_returns_dense_strategy() -> None:
    prediction = {
        "answers": {
            "retrieval_strategy": {
                "type": "choice",
                "choice": "dense",
                "probabilities": {
                    "dense": 0.8,
                    "bm25": 0.1,
                    "hybrid": 0.1,
                },
                "confidence": 0.5,
                "answer_confidence": 0.8,
            }
        }
    }

    with patch("lectra.infrastructure.routing.laya.Router") as router_class:
        router_class.return_value.predict.return_value = prediction

        router = LayaQueryRouter()

        decision = router.route("Explain how virtual memory works.")

    assert decision.strategy is RetrievalStrategy.DENSE


def test_laya_router_returns_bm25_strategy() -> None:
    prediction = {
        "answers": {
            "retrieval_strategy": {
                "type": "choice",
                "choice": "bm25",
                "probabilities": {
                    "dense": 0.1,
                    "bm25": 0.8,
                    "hybrid": 0.1,
                },
                "confidence": 0.5,
                "answer_confidence": 0.8,
            }
        }
    }

    with patch("lectra.infrastructure.routing.laya.Router") as router_class:
        router_class.return_value.predict.return_value = prediction

        router = LayaQueryRouter()

        decision = router.route("What is the FIFO page replacement algorithm?")

    assert decision.strategy is RetrievalStrategy.BM25


def test_laya_router_returns_hybrid_strategy() -> None:
    prediction = {
        "answers": {
            "retrieval_strategy": {
                "type": "choice",
                "choice": "hybrid",
                "probabilities": {
                    "dense": 0.1,
                    "bm25": 0.1,
                    "hybrid": 0.8,
                },
                "confidence": 0.5,
                "answer_confidence": 0.8,
            }
        }
    }

    with patch("lectra.infrastructure.routing.laya.Router") as router_class:
        router_class.return_value.predict.return_value = prediction

        router = LayaQueryRouter()

        decision = router.route("Compare FIFO and LRU page replacement.")

    assert decision.strategy is RetrievalStrategy.HYBRID


def test_laya_router_rejects_empty_query() -> None:
    with patch("lectra.infrastructure.routing.laya.Router"):
        router = LayaQueryRouter()

    with pytest.raises(ValueError):
        router.route("   ")
