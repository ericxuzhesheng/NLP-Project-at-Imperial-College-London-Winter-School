"""Skip-gram word representation models."""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def train_gensim_word2vec(
    sentences: Iterable[Iterable[str] | str],
    output_path: str | Path,
    vector_size: int = 256,
    window: int = 5,
    negative: int = 10,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 10,
) -> None:
    """Train Gensim skip-gram with negative sampling, matching the notebook track."""

    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise RuntimeError("Install gensim to train Gensim Word2Vec.") from exc

    corpus = _normalize(sentences)
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        sg=1,
        negative=negative,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))


@dataclass
class SkipGramNegativeSampling:
    """A compact NumPy skip-gram with negative sampling implementation."""

    vector_size: int = 100
    window_size: int = 5
    negative_samples: int = 5
    max_vocab: int = 10000
    min_count: int = 2
    learning_rate: float = 0.025
    epochs: int = 1
    seed: int = 1022
    vocabulary: list[str] | None = None
    vectors: np.ndarray | None = None
    context_vectors: np.ndarray | None = None
    negative_distribution: np.ndarray | None = None

    def fit(self, sentences: Iterable[Iterable[str] | str]) -> "SkipGramNegativeSampling":
        rng = np.random.default_rng(self.seed)
        corpus = _normalize(sentences)
        counts = Counter(token for sentence in corpus for token in sentence)
        words = [word for word, count in counts.most_common(self.max_vocab) if count >= self.min_count]
        if not words:
            raise ValueError("No vocabulary terms survived min_count filtering.")

        self.vocabulary = words
        word_to_id = {word: index for index, word in enumerate(words)}
        vocab_size = len(words)
        scale = 0.5 / max(1, self.vector_size)
        self.vectors = rng.uniform(-scale, scale, (vocab_size, self.vector_size)).astype(np.float32)
        self.context_vectors = np.zeros((vocab_size, self.vector_size), dtype=np.float32)
        frequencies = np.array([counts[word] for word in words], dtype=np.float64) ** 0.75
        self.negative_distribution = frequencies / frequencies.sum()

        pairs = self._training_pairs(corpus, word_to_id)
        if not pairs:
            raise ValueError("No training pairs found. Increase corpus size or window size.")

        for epoch in range(self.epochs):
            rng.shuffle(pairs)
            lr = self.learning_rate * (1.0 - epoch / max(1, self.epochs))
            for center_id, context_id in pairs:
                self._update(center_id, context_id, 1.0, lr)
                negatives = rng.choice(
                    vocab_size,
                    size=self.negative_samples,
                    replace=True,
                    p=self.negative_distribution,
                )
                for negative_id in negatives:
                    if negative_id != context_id:
                        self._update(center_id, int(negative_id), 0.0, lr)
        return self

    @property
    def word_to_id(self) -> dict[str, int]:
        if self.vocabulary is None:
            return {}
        return {word: index for index, word in enumerate(self.vocabulary)}

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        if self.vectors is None or self.vocabulary is None:
            raise ValueError("Model has not been fitted.")
        if word not in self.word_to_id:
            raise KeyError(f"{word!r} is not in the vocabulary.")
        target = self.vectors[self.word_to_id[word]]
        scores = (self.vectors @ target) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(target) + 1e-12
        )
        rows = [
            (candidate, float(scores[index]))
            for index, candidate in enumerate(self.vocabulary)
            if candidate != word
        ]
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:topn]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "SkipGramNegativeSampling":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}.")
        return model

    def _training_pairs(
        self,
        corpus: list[list[str]],
        word_to_id: dict[str, int],
    ) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for sentence in corpus:
            ids = [word_to_id[token] for token in sentence if token in word_to_id]
            for center_pos, center_id in enumerate(ids):
                left = max(0, center_pos - self.window_size)
                right = min(len(ids), center_pos + self.window_size + 1)
                for context_pos in range(left, right):
                    if context_pos != center_pos:
                        pairs.append((center_id, ids[context_pos]))
        return pairs

    def _update(self, center_id: int, context_id: int, label: float, learning_rate: float) -> None:
        assert self.vectors is not None
        assert self.context_vectors is not None
        center = self.vectors[center_id].copy()
        context = self.context_vectors[context_id].copy()
        score = _sigmoid(float(center @ context))
        gradient = learning_rate * (label - score)
        self.vectors[center_id] += gradient * context
        self.context_vectors[context_id] += gradient * center


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = np.exp(-value)
        return float(1 / (1 + z))
    z = np.exp(value)
    return float(z / (1 + z))


def _normalize(sentences: Iterable[Iterable[str] | str]) -> list[list[str]]:
    corpus: list[list[str]] = []
    for sentence in sentences:
        tokens = sentence.split() if isinstance(sentence, str) else list(sentence)
        tokens = [token for token in tokens if token]
        if tokens:
            corpus.append(tokens)
    return corpus
