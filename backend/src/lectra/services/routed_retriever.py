from lectra.domain.retrieval import RetrievedChunk
from lectra.domain.retrieval_strategy import RetrievalStrategy
from lectra.services.retriever import Retriever
from lectra.services.router import QueryRouter


class RoutedRetriever:
    """Retrieves chunks using a strategy selected by a query router."""

    def __init__(
        self,
        router: QueryRouter,
        retrievers: dict[RetrievalStrategy, Retriever],
    ) -> None:
        self._router = router
        self._retrievers = retrievers

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Route a query and delegate retrieval to the selected retriever."""

        decision = self._router.route(query)

        try:
            retriever = self._retrievers[decision.strategy]
        except KeyError as exc:
            raise ValueError(
                f"No retriever configured for strategy: {decision.strategy}"
            ) from exc

        return retriever.retrieve(
            query,
            top_k=top_k,
        )
