from typing import Protocol

from lectra.domain.routing import RetrievalDecision


class QueryRouter(Protocol):
    """Defines the query-routing capability required by Lectra."""

    def route(self, query: str) -> RetrievalDecision:
        """Select a retrieval strategy for a query."""
        ...
