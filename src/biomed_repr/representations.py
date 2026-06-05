"""Word representation models."""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _as_sentences(sentences: Iterable[Iterable[str] | str]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for sentence in sentences:
        if isinstance(sentence, str):
            tokens = sentence.split()
        else:
            tokens = list(sentence)
        tokens = [token for token in tokens if token]
        if tokens:
            normalized.append(tokens)
    return normalized


@dataclass
class CoOccurrenceEmbeddings:
    """Build word vectors from a shifted positive PMI co-occurrence matrix."""

    vector_size: int = 100
    window_size: int = 5
    max_vocab: int = 10000
    min_count: int = 2
    shift: float = 1.0
    vocabulary: list[str] | None = None
    vectors: np.ndarray | None = None

    def fit(self, sentences: Iterable[Iterable[str] | str]) -> "CoOccurrenceEmbeddings":
        corpus = _as_sentences(sentences)
        counts = Counter(token for sentence in corpus for token in sentence)
        words = [
            word
            for word, count in counts.most_common(self.max_vocab)
            if count >= self.min_count
        ]
        if not words:
            raise ValueError("No vocabulary terms survived min_count filtering.")

        self.vocabulary = words
        word_to_id = {word: index for index, word in enumerate(words)}
        pair_counts: dict[tuple[int, int], float] = defaultdict(float)
        row_totals = np.zeros(len(words), dtype=np.float64)
        col_totals = np.zeros(len(words), dtype=np.float64)

        for sentence in corpus:
            ids = [word_to_id[token] for token in sentence if token in word_to_id]
            for center_pos, center_id in enumerate(ids):
                left = max(0, center_pos - self.window_size)
                right = min(len(ids), center_pos + self.window_size + 1)
                for context_pos in range(left, right):
                    if context_pos == center_pos:
                        continue
                    context_id = ids[context_pos]
                    weight = 1.0 / abs(context_pos - center_pos)
                    pair_counts[(center_id, context_id)] += weight
                    row_totals[center_id] += weight
                    col_totals[context_id] += weight

        matrix = np.zeros((len(words), len(words)), dtype=np.float32)
        total = float(row_totals.sum())
        if total == 0:
            raise ValueError("No co-occurrences found. Increase corpus size or window size.")

        log_shift = math.log(self.shift)
        for (row, col), value in pair_counts.items():
            pmi = math.log(value * total / (row_totals[row] * col_totals[col])) - log_shift
            if pmi > 0:
                matrix[row, col] = pmi

        components = min(self.vector_size, max(1, len(words) - 1))
        _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
        self.vectors = right_vectors[:components].T * np.sqrt(singular_values[:components])
        return self

    @property
    def word_to_id(self) -> dict[str, int]:
        if self.vocabulary is None:
            return {}
        return {word: index for index, word in enumerate(self.vocabulary)}

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_id

    def vector(self, word: str) -> np.ndarray:
        if self.vectors is None or self.vocabulary is None:
            raise ValueError("Model has not been fitted.")
        return self.vectors[self.word_to_id[word]]

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        if self.vectors is None or self.vocabulary is None:
            raise ValueError("Model has not been fitted.")
        if word not in self.word_to_id:
            raise KeyError(f"{word!r} is not in the vocabulary.")

        vectors = self.vectors
        target = self.vector(word)
        target_norm = np.linalg.norm(target)
        norms = np.linalg.norm(vectors, axis=1)
        scores = (vectors @ target) / (norms * target_norm + 1e-12)
        scored = [
            (candidate, float(scores[index]))
            for index, candidate in enumerate(self.vocabulary)
            if candidate != word
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:topn]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "CoOccurrenceEmbeddings":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}.")
        return model
