"""Corpus loading utilities for biomedical text."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Document:
    """A lightweight text document extracted from CORD-19 style sources."""

    doc_id: str
    title: str
    text: str

    @property
    def combined_text(self) -> str:
        parts = [self.title.strip(), self.text.strip()]
        return "\n".join(part for part in parts if part)


def iter_metadata_csv(path: str | Path, limit: int | None = None) -> Iterator[Document]:
    """Yield title and abstract text from a CORD-19 metadata CSV file."""

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            if not title and not abstract:
                continue
            doc_id = row.get("cord_uid") or row.get("sha") or str(index)
            yield Document(doc_id=doc_id, title=title, text=abstract)


def iter_cord19_json(root: str | Path, limit: int | None = None) -> Iterator[Document]:
    """Yield title, abstract, and body text from CORD-19 parsed JSON files."""

    root = Path(root)
    count = 0
    for json_path in sorted(root.rglob("*.json")):
        if limit is not None and count >= limit:
            break
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata") or {}
        title = (metadata.get("title") or "").strip()
        paragraphs: list[str] = []
        for section in ("abstract", "body_text"):
            for item in payload.get(section) or []:
                text = (item.get("text") or "").strip()
                if text:
                    paragraphs.append(text)

        if title or paragraphs:
            count += 1
            yield Document(
                doc_id=payload.get("paper_id") or json_path.stem,
                title=title,
                text="\n".join(paragraphs),
            )


def iter_plain_text(path: str | Path, limit: int | None = None) -> Iterator[Document]:
    """Yield one document per non-empty line from a plain text corpus."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            text = line.strip()
            if text:
                yield Document(doc_id=str(index), title="", text=text)


def write_text_corpus(documents: Iterable[Document], output_path: str | Path) -> int:
    """Write documents into a blank-line separated UTF-8 text corpus."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            text = document.combined_text.strip()
            if not text:
                continue
            handle.write(text)
            handle.write("\n\n")
            count += 1
    return count
