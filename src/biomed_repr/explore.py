"""Exploration helpers for biomedical word vectors."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np


class SimilarityModel(Protocol):
    vocabulary: list[str] | None
    vectors: np.ndarray | None

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        ...


BIOMEDICAL_ENTITIES: dict[str, list[str]] = {
    "virus": ["coronavirus", "sars-cov-2", "influenza", "hiv", "norovirus"],
    "disease": ["covid-19", "pneumonia", "cancer", "asthma", "diabetes"],
    "symptom": ["fever", "cough", "fatigue", "headache", "dyspnea"],
    "protein": ["ace2", "antibody", "cytokine", "enzyme", "receptor"],
    "treatment": ["vaccine", "antiviral", "remdesivir", "chloroquine", "therapy"],
}


def write_similar_words(
    model: SimilarityModel,
    target_word: str,
    output_path: str | Path,
    topn: int = 10,
) -> list[tuple[str, float]]:
    """Write nearest neighbors for a target word to CSV."""

    rows = model.most_similar(target_word, topn=topn)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word", "similarity"])
        writer.writerows(rows)
    return rows


def available_entities(
    vocabulary: list[str],
    entity_groups: Mapping[str, list[str]] = BIOMEDICAL_ENTITIES,
) -> dict[str, list[str]]:
    """Return configured biomedical entities that appear in a vocabulary."""

    vocab = set(vocabulary)
    return {
        group: [entity for entity in entities if entity in vocab]
        for group, entities in entity_groups.items()
    }


def save_tsne_plot(
    vocabulary: list[str],
    vectors: np.ndarray,
    output_path: str | Path,
    max_words: int = 300,
    seed: int = 1022,
) -> None:
    """Save a t-SNE visualization for fitted vectors."""

    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("Install matplotlib and scikit-learn to plot t-SNE.") from exc

    n_words = min(max_words, len(vocabulary))
    if n_words < 2:
        raise ValueError("At least two words are required for t-SNE.")

    selected_words = vocabulary[:n_words]
    selected_vectors = vectors[:n_words]
    perplexity = min(30, max(1, n_words - 1))
    points = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(selected_vectors)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.scatter(points[:, 0], points[:, 1], s=12, color="#376996", alpha=0.75)
    for word, (x_coord, y_coord) in zip(selected_words, points):
        plt.text(x_coord, y_coord, word, fontsize=7)
    plt.title("t-SNE Visualization of Biomedical Word Embeddings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
