"""Command line interface for the biomedical representation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .corpus import iter_cord19_json, iter_metadata_csv, iter_plain_text, write_text_corpus
from .explore import save_tsne_plot, write_similar_words
from .representations import CoOccurrenceEmbeddings
from .tokenization import tokenize_file


def _read_tokenized(path: Path, limit: int | None = None) -> list[list[str]]:
    sentences: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            tokens = line.split()
            if tokens:
                sentences.append(tokens)
    return sentences


def prepare_corpus(args: argparse.Namespace) -> None:
    if args.source_type == "metadata":
        documents = iter_metadata_csv(args.input, limit=args.limit)
    elif args.source_type == "cord19-json":
        documents = iter_cord19_json(args.input, limit=args.limit)
    else:
        documents = iter_plain_text(args.input, limit=args.limit)
    count = write_text_corpus(documents, args.output)
    print(f"Wrote {count} documents to {args.output}")


def tokenize(args: argparse.Namespace) -> None:
    count = tokenize_file(
        args.input,
        args.output,
        tokenizer=args.tokenizer,
        lowercase=not args.keep_case,
        model_name=args.model_name,
        limit=args.limit,
    )
    print(f"Tokenized {count} lines to {args.output}")


def train(args: argparse.Namespace) -> None:
    sentences = _read_tokenized(args.input, limit=args.limit)
    model = CoOccurrenceEmbeddings(
        vector_size=args.vector_size,
        window_size=args.window_size,
        max_vocab=args.max_vocab,
        min_count=args.min_count,
    ).fit(sentences)
    model.save(args.output)
    print(f"Saved {len(model.vocabulary or [])} word vectors to {args.output}")


def similar(args: argparse.Namespace) -> None:
    model = CoOccurrenceEmbeddings.load(args.model)
    rows = write_similar_words(model, args.word, args.output, topn=args.topn)
    for word, score in rows:
        print(f"{word}\t{score:.4f}")


def plot(args: argparse.Namespace) -> None:
    model = CoOccurrenceEmbeddings.load(args.model)
    if model.vocabulary is None or model.vectors is None:
        raise ValueError("Model has not been fitted.")
    save_tsne_plot(model.vocabulary, model.vectors, args.output, max_words=args.max_words)
    print(f"Saved t-SNE plot to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biomed-repr",
        description="Biomedical word representation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus_parser = subparsers.add_parser("prepare-corpus")
    corpus_parser.add_argument("--input", type=Path, required=True)
    corpus_parser.add_argument("--output", type=Path, required=True)
    corpus_parser.add_argument(
        "--source-type",
        choices=["metadata", "cord19-json", "plain-text"],
        default="plain-text",
    )
    corpus_parser.add_argument("--limit", type=int)
    corpus_parser.set_defaults(func=prepare_corpus)

    tokenize_parser = subparsers.add_parser("tokenize")
    tokenize_parser.add_argument("--input", type=Path, required=True)
    tokenize_parser.add_argument("--output", type=Path, required=True)
    tokenize_parser.add_argument("--tokenizer", choices=["regex", "nltk", "bert"], default="regex")
    tokenize_parser.add_argument("--model-name", default="bert-base-uncased")
    tokenize_parser.add_argument("--keep-case", action="store_true")
    tokenize_parser.add_argument("--limit", type=int)
    tokenize_parser.set_defaults(func=tokenize)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--vector-size", type=int, default=100)
    train_parser.add_argument("--window-size", type=int, default=5)
    train_parser.add_argument("--max-vocab", type=int, default=10000)
    train_parser.add_argument("--min-count", type=int, default=2)
    train_parser.add_argument("--limit", type=int)
    train_parser.set_defaults(func=train)

    similar_parser = subparsers.add_parser("similar")
    similar_parser.add_argument("--model", type=Path, required=True)
    similar_parser.add_argument("--word", required=True)
    similar_parser.add_argument("--output", type=Path, default=Path("outputs/similar_words.csv"))
    similar_parser.add_argument("--topn", type=int, default=10)
    similar_parser.set_defaults(func=similar)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--model", type=Path, required=True)
    plot_parser.add_argument("--output", type=Path, default=Path("outputs/tsne.png"))
    plot_parser.add_argument("--max-words", type=int, default=300)
    plot_parser.set_defaults(func=plot)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
