"""N-gram language model and co-occurrence vectors."""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .representations import CoOccurrenceEmbeddings


@dataclass
class NGramLanguageModel:
    """Smoothed N-gram model with companion co-occurrence word vectors."""

    n: int = 3
    max_vocab: int = 10000
    min_count: int = 2
    alpha: float = 0.1
    vector_size: int = 100
    vocabulary: list[str] | None = None
    language_vocabulary: list[str] | None = None
    ngram_counts: dict[tuple[str, ...], Counter[str]] | None = None
    context_counts: Counter[tuple[str, ...]] | None = None
    embeddings: CoOccurrenceEmbeddings | None = None

    def fit(self, sentences: Iterable[Iterable[str] | str]) -> "NGramLanguageModel":
        corpus = _normalize(sentences)
        counts = Counter(token for sentence in corpus for token in sentence)
        words = [word for word, count in counts.most_common(self.max_vocab) if count >= self.min_count]
        specials = ["<UNK>", "<START>", "<END>"]
        self.language_vocabulary = specials + [word for word in words if word not in specials]
        vocab = set(self.language_vocabulary)
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()

        mapped_corpus: list[list[str]] = []
        for sentence in corpus:
            mapped = [token if token in vocab else "<UNK>" for token in sentence]
            padded = ["<START>"] * (self.n - 1) + mapped + ["<END>"]
            mapped_corpus.append(mapped)
            for index in range(self.n - 1, len(padded)):
                context = tuple(padded[index - self.n + 1 : index])
                target = padded[index]
                self.ngram_counts[context][target] += 1
                self.context_counts[context] += 1

        self.embeddings = CoOccurrenceEmbeddings(
            vector_size=self.vector_size,
            window_size=max(1, self.n - 1),
            max_vocab=self.max_vocab,
            min_count=self.min_count,
        ).fit(mapped_corpus)
        self.vocabulary = self.embeddings.vocabulary
        return self

    def predict_next(self, context: Iterable[str], topn: int = 10) -> list[tuple[str, float]]:
        if (
            self.ngram_counts is None
            or self.context_counts is None
            or self.language_vocabulary is None
        ):
            raise ValueError("Model has not been fitted.")
        vocab = set(self.language_vocabulary)
        context_tokens = [token if token in vocab else "<UNK>" for token in context]
        context_tuple = tuple(context_tokens[-(self.n - 1) :])
        if len(context_tuple) < self.n - 1:
            context_tuple = ("<START>",) * (self.n - 1 - len(context_tuple)) + context_tuple
        counts = self.ngram_counts.get(context_tuple, Counter())
        denominator = self.context_counts.get(context_tuple, 0) + self.alpha * len(
            self.language_vocabulary
        )
        scored = [
            (word, (counts.get(word, 0) + self.alpha) / denominator)
            for word in self.language_vocabulary
            if word != "<START>"
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:topn]

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        if self.embeddings is None:
            raise ValueError("Model has not been fitted.")
        return self.embeddings.most_similar(word, topn=topn)

    @property
    def vectors(self) -> np.ndarray | None:
        if self.embeddings is None:
            return None
        return self.embeddings.vectors

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "NGramLanguageModel":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}.")
        return model


def _normalize(sentences: Iterable[Iterable[str] | str]) -> list[list[str]]:
    corpus: list[list[str]] = []
    for sentence in sentences:
        tokens = sentence.split() if isinstance(sentence, str) else list(sentence)
        tokens = [token for token in tokens if token]
        if tokens:
            corpus.append(tokens)
    return corpus
