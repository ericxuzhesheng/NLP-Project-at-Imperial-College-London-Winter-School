"""Tokenization strategies for biomedical corpora."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal

TOKEN_RE = re.compile(r"[A-Za-z]+[A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*|\d+(?:\.\d+)?")
TokenizerName = Literal["regex", "nltk", "bert"]


def regex_tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize text with a deterministic regex that keeps biomedical hyphens."""

    tokens = TOKEN_RE.findall(text)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def nltk_tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize with NLTK when punkt resources are available."""

    try:
        from nltk import word_tokenize
    except ImportError as exc:
        raise RuntimeError("Install nltk to use the NLTK tokenizer.") from exc

    tokens = [token for token in word_tokenize(text) if TOKEN_RE.fullmatch(token)]
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def bert_tokenize(
    text: str,
    model_name: str = "bert-base-uncased",
    lowercase: bool = True,
) -> list[str]:
    """Tokenize with a Hugging Face tokenizer."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to use the BERT tokenizer.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def tokenize_text(
    text: str,
    tokenizer: TokenizerName = "regex",
    lowercase: bool = True,
    model_name: str = "bert-base-uncased",
) -> list[str]:
    """Dispatch to a named tokenizer."""

    if tokenizer == "regex":
        return regex_tokenize(text, lowercase=lowercase)
    if tokenizer == "nltk":
        return nltk_tokenize(text, lowercase=lowercase)
    if tokenizer == "bert":
        return bert_tokenize(text, model_name=model_name, lowercase=lowercase)
    raise ValueError(f"Unsupported tokenizer: {tokenizer}")


def tokenize_lines(
    lines: Iterable[str],
    tokenizer: TokenizerName = "regex",
    lowercase: bool = True,
    model_name: str = "bert-base-uncased",
) -> list[list[str]]:
    """Tokenize an iterable of documents or sentences."""

    return [
        tokenize_text(line, tokenizer=tokenizer, lowercase=lowercase, model_name=model_name)
        for line in lines
        if line.strip()
    ]


def tokenize_file(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: TokenizerName = "regex",
    lowercase: bool = True,
    model_name: str = "bert-base-uncased",
    limit: int | None = None,
) -> int:
    """Tokenize a text file and write one tokenized line per input line."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if limit is not None and count >= limit:
                break
            tokens = tokenize_text(
                line,
                tokenizer=tokenizer,
                lowercase=lowercase,
                model_name=model_name,
            )
            if not tokens:
                continue
            target.write(" ".join(tokens))
            target.write("\n")
            count += 1
    return count
