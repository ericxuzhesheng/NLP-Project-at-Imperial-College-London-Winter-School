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
    """Write nearest neighbors for a target word to CSV or XLSX."""

    rows = model.most_similar(target_word, topn=topn)
    _write_table(output_path, ["word", "similarity"], rows)
    return rows


def _write_table(
    output_path: str | Path,
    headers: list[str],
    rows: list[tuple[str, float]] | list[tuple[str, str, float, float]],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".xlsx":
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            raise RuntimeError("Install openpyxl to write XLSX outputs.") from exc
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(headers)
        for row in rows:
            sheet.append(list(row))
        workbook.save(output_path)
    else:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)


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


def save_entity_tsne_plot(
    vocabulary: list[str],
    vectors: np.ndarray,
    output_path: str | Path,
    entity_groups: Mapping[str, list[str]] = BIOMEDICAL_ENTITIES,
    max_background_words: int = 500,
    seed: int = 1022,
) -> None:
    """Save a t-SNE plot with biomedical entity groups highlighted by color."""

    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("Install matplotlib and scikit-learn to plot t-SNE.") from exc

    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    entity_rows: list[tuple[str, str, int]] = []
    seen_entities: set[str] = set()
    for group, entities in entity_groups.items():
        for entity in entities:
            if entity in word_to_id and entity not in seen_entities:
                entity_rows.append((entity, group, word_to_id[entity]))
                seen_entities.add(entity)

    background = [
        (word, "background", index)
        for index, word in enumerate(vocabulary[:max_background_words])
        if word not in seen_entities
    ]
    rows = entity_rows + background
    if len(rows) < 2:
        raise ValueError("At least two vocabulary terms are required for t-SNE.")

    selected_vectors = np.array([vectors[index] for _, _, index in rows])
    perplexity = min(30, max(1, len(rows) - 1))
    points = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(selected_vectors)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    palette = {
        "virus": "#d73027",
        "disease": "#4575b4",
        "symptom": "#fdae61",
        "protein": "#1a9850",
        "treatment": "#984ea3",
        "background": "#9e9e9e",
    }
    plt.figure(figsize=(12, 8))
    for group in ["background", *entity_groups.keys()]:
        indices = [index for index, (_, row_group, _) in enumerate(rows) if row_group == group]
        if not indices:
            continue
        size = 8 if group == "background" else 36
        alpha = 0.25 if group == "background" else 0.85
        plt.scatter(
            points[indices, 0],
            points[indices, 1],
            s=size,
            color=palette.get(group, "#333333"),
            label=group,
            alpha=alpha,
        )
    for index, (word, group, _) in enumerate(rows):
        if group != "background":
            plt.text(points[index, 0], points[index, 1], word, fontsize=8)
    plt.title("Biomedical Entity t-SNE Visualization")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_entity_inventory(
    vocabulary: list[str],
    output_path: str | Path,
    entity_groups: Mapping[str, list[str]] = BIOMEDICAL_ENTITIES,
) -> dict[str, list[str]]:
    """Write the biomedical entities found in a vocabulary."""

    found = available_entities(vocabulary, entity_groups=entity_groups)
    rows: list[tuple[str, str, float, float]] = []
    for group, words in found.items():
        for word in words:
            rows.append((group, word, 1.0, 0.0))
    _write_table(output_path, ["group", "word", "present", "reserved"], rows)
    return found
