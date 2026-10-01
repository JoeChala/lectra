import math
import re
from collections import Counter

from lectra.domain.retrieval import Chunk, RetrievedChunk


class BM25Retriever:
    """Retrieves chunks using lexical BM25 scoring."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._chunks = chunks
        self._k1 = k1
        self._b = b

        self._tokenized_chunks = [self._tokenize(chunk.content) for chunk in chunks]

        self._document_frequencies = self._build_document_frequencies()
        self._average_document_length = sum(
            len(tokens) for tokens in self._tokenized_chunks
        ) / len(self._tokenized_chunks)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks using BM25."""

        query_tokens = self._tokenize(query)
        scores = [
            self._score(query_tokens, tokens) for tokens in self._tokenized_chunks
        ]

        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            RetrievedChunk(
                chunk=self._chunks[index],
                score=scores[index],
            )
            for index in ranked_indices
        ]

    def _score(
        self,
        query_tokens: list[str],
        document_tokens: list[str],
    ) -> float:
        """Calculate the BM25 score for one document."""

        document_length = len(document_tokens)
        term_counts = Counter(document_tokens)

        score = 0.0

        for token in query_tokens:
            frequency = term_counts[token]

            if frequency == 0:
                continue

            document_frequency = self._document_frequencies.get(token, 0)

            idf = math.log(
                1
                + (len(self._chunks) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            numerator = frequency * (self._k1 + 1)

            denominator = frequency + self._k1 * (
                1 - self._b + self._b * document_length / self._average_document_length
            )

            score += idf * numerator / denominator

        return score

    def _build_document_frequencies(self) -> dict[str, int]:
        """Build document frequencies for the indexed chunks."""

        frequencies: dict[str, int] = {}

        for tokens in self._tokenized_chunks:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1

        return frequencies

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into normalized lexical terms."""

        return re.findall(r"\b\w+\b", text.lower())
