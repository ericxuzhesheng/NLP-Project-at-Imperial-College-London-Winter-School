"""Byte-pair encoding tokenizers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .tokenization import regex_tokenize

END_TOKEN = "</w>"


def train_huggingface_bpe(
    input_path: str | Path,
    output_path: str | Path,
    vocab_size: int = 30000,
    min_frequency: int = 2,
) -> None:
    """Train a Hugging Face tokenizers BPE model, matching the notebook track."""

    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise RuntimeError("Install tokenizers to train Hugging Face BPE.") from exc

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
    )
    tokenizer.train(files=[str(input_path)], trainer=trainer)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))


def tokenize_huggingface_bpe(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    """Tokenize a text file with a saved Hugging Face tokenizers BPE model."""

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install tokenizers to use Hugging Face BPE.") from exc

    tokenizer = Tokenizer.from_file(str(model_path))
    return _tokenize_with_callable(
        input_path,
        output_path,
        lambda text: tokenizer.encode(text).tokens,
        limit=limit,
    )


@dataclass
class SimpleBPE:
    """A small from-scratch BPE tokenizer for reproducible coursework experiments."""

    merges: list[tuple[str, str]]
    vocab: dict[str, int]
    lowercase: bool = True

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        num_merges: int = 200,
        min_frequency: int = 2,
        lowercase: bool = True,
    ) -> "SimpleBPE":
        word_counts: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for token in regex_tokenize(text, lowercase=lowercase):
                word_counts[tuple(token) + (END_TOKEN,)] += 1

        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, count in word_counts.items():
                for left, right in zip(word, word[1:]):
                    pair_counts[(left, right)] += count
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break
            merges.append(best_pair)
            word_counts = _merge_pair_counts(word_counts, best_pair)

        vocab: Counter[str] = Counter()
        for word, count in word_counts.items():
            for symbol in word:
                if symbol != END_TOKEN:
                    vocab[symbol] += count
        return cls(merges=merges, vocab=dict(vocab), lowercase=lowercase)

    def encode_word(self, word: str) -> list[str]:
        symbols = tuple(word.lower() if self.lowercase else word) + (END_TOKEN,)
        for pair in self.merges:
            symbols = _merge_pair(symbols, pair)
        return [symbol for symbol in symbols if symbol != END_TOKEN]

    def encode(self, text: str) -> list[str]:
        pieces: list[str] = []
        for token in regex_tokenize(text, lowercase=self.lowercase):
            pieces.extend(self.encode_word(token))
        return pieces

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "simple_bpe",
            "lowercase": self.lowercase,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleBPE":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(
            merges=merges,
            vocab=dict(payload["vocab"]),
            lowercase=bool(payload.get("lowercase", True)),
        )


def train_simple_bpe_file(
    input_path: str | Path,
    output_path: str | Path,
    num_merges: int = 200,
    min_frequency: int = 2,
    limit: int | None = None,
) -> SimpleBPE:
    lines = _read_lines(input_path, limit=limit)
    tokenizer = SimpleBPE.train(lines, num_merges=num_merges, min_frequency=min_frequency)
    tokenizer.save(output_path)
    return tokenizer


def tokenize_simple_bpe_file(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    tokenizer = SimpleBPE.load(model_path)
    return _tokenize_with_callable(input_path, output_path, tokenizer.encode, limit=limit)


def _read_lines(path: str | Path, limit: int | None = None) -> list[str]:
    lines: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                lines.append(line.strip())
    return lines


def _tokenize_with_callable(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer,
    limit: int | None = None,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(input_path).open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if limit is not None and count >= limit:
                break
            tokens = tokenizer(line)
            if tokens:
                target.write(" ".join(tokens))
                target.write("\n")
                count += 1
    return count


def _merge_pair_counts(
    word_counts: Counter[tuple[str, ...]],
    pair: tuple[str, str],
) -> Counter[tuple[str, ...]]:
    merged: Counter[tuple[str, ...]] = Counter()
    for word, count in word_counts.items():
        merged[_merge_pair(word, pair)] += count
    return merged


def _merge_pair(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    output: list[str] = []
    index = 0
    while index < len(word):
        if index < len(word) - 1 and (word[index], word[index + 1]) == pair:
            output.append(word[index] + word[index + 1])
            index += 2
        else:
            output.append(word[index])
            index += 1
    return tuple(output)
