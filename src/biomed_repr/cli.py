"""Command line interface for the biomedical representation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bpe import (
    tokenize_huggingface_bpe,
    tokenize_simple_bpe_file,
    train_huggingface_bpe,
    train_simple_bpe_file,
)
from .corpus import iter_cord19_json, iter_metadata_csv, iter_plain_text, write_text_corpus
from .explore import save_entity_tsne_plot, save_tsne_plot, write_entity_inventory, write_similar_words
from .mlm import train_mlm_lora
from .ngram import NGramLanguageModel
from .representations import CoOccurrenceEmbeddings
from .tokenization import tokenize_file
from .word2vec import SkipGramNegativeSampling, train_gensim_word2vec


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
    if args.model_type == "cooccurrence":
        model = CoOccurrenceEmbeddings(
            vector_size=args.vector_size,
            window_size=args.window_size,
            max_vocab=args.max_vocab,
            min_count=args.min_count,
        ).fit(sentences)
    elif args.model_type == "ngram":
        model = NGramLanguageModel(
            n=args.n,
            max_vocab=args.max_vocab,
            min_count=args.min_count,
            vector_size=args.vector_size,
        ).fit(sentences)
    elif args.model_type == "sgns":
        model = SkipGramNegativeSampling(
            vector_size=args.vector_size,
            window_size=args.window_size,
            negative_samples=args.negative_samples,
            max_vocab=args.max_vocab,
            min_count=args.min_count,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            seed=args.seed,
        ).fit(sentences)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    model.save(args.output)
    print(f"Saved {len(model.vocabulary or [])} word vectors to {args.output}")


def similar(args: argparse.Namespace) -> None:
    model = _load_similarity_model(args.model, args.model_type)
    rows = write_similar_words(model, args.word, args.output, topn=args.topn)
    for word, score in rows:
        print(f"{word}\t{score:.4f}")


def plot(args: argparse.Namespace) -> None:
    model = _load_similarity_model(args.model, args.model_type)
    if model.vocabulary is None or model.vectors is None:
        raise ValueError("Model has not been fitted.")
    if args.entities:
        save_entity_tsne_plot(
            model.vocabulary,
            model.vectors,
            args.output,
            max_background_words=args.max_words,
        )
    else:
        save_tsne_plot(model.vocabulary, model.vectors, args.output, max_words=args.max_words)
    print(f"Saved t-SNE plot to {args.output}")


def bpe_train(args: argparse.Namespace) -> None:
    if args.backend == "simple":
        tokenizer = train_simple_bpe_file(
            args.input,
            args.output,
            num_merges=args.num_merges,
            min_frequency=args.min_frequency,
            limit=args.limit,
        )
        print(f"Saved simple BPE with {len(tokenizer.merges)} merges to {args.output}")
    else:
        train_huggingface_bpe(
            args.input,
            args.output,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        print(f"Saved Hugging Face BPE tokenizer to {args.output}")


def bpe_tokenize(args: argparse.Namespace) -> None:
    if args.backend == "simple":
        count = tokenize_simple_bpe_file(args.model, args.input, args.output, limit=args.limit)
    else:
        count = tokenize_huggingface_bpe(args.model, args.input, args.output, limit=args.limit)
    print(f"BPE-tokenized {count} lines to {args.output}")


def train_word2vec(args: argparse.Namespace) -> None:
    sentences = _read_tokenized(args.input, limit=args.limit)
    train_gensim_word2vec(
        sentences,
        args.output,
        vector_size=args.vector_size,
        window=args.window_size,
        negative=args.negative_samples,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
    )
    print(f"Saved Gensim Word2Vec model to {args.output}")


def train_mlm(args: argparse.Namespace) -> None:
    train_mlm_lora(
        args.input,
        args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        max_lines=args.max_lines,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    print(f"Saved MLM-LoRA model to {args.output_dir}")


def predict_next(args: argparse.Namespace) -> None:
    model = NGramLanguageModel.load(args.model)
    rows = model.predict_next(args.context.split(), topn=args.topn)
    for word, probability in rows:
        print(f"{word}\t{probability:.6f}")


def entity_inventory(args: argparse.Namespace) -> None:
    model = _load_similarity_model(args.model, args.model_type)
    if model.vocabulary is None:
        raise ValueError("Model has not been fitted.")
    found = write_entity_inventory(model.vocabulary, args.output)
    for group, words in found.items():
        print(f"{group}: {', '.join(words) if words else '(none)'}")


def _load_similarity_model(path: Path, model_type: str):
    if model_type == "cooccurrence":
        return CoOccurrenceEmbeddings.load(path)
    if model_type == "ngram":
        return NGramLanguageModel.load(path)
    if model_type == "sgns":
        return SkipGramNegativeSampling.load(path)
    raise ValueError(f"Unsupported model type: {model_type}")


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
    train_parser.add_argument(
        "--model-type",
        choices=["cooccurrence", "ngram", "sgns"],
        default="cooccurrence",
    )
    train_parser.add_argument("--n", type=int, default=3)
    train_parser.add_argument("--vector-size", type=int, default=100)
    train_parser.add_argument("--window-size", type=int, default=5)
    train_parser.add_argument("--max-vocab", type=int, default=10000)
    train_parser.add_argument("--min-count", type=int, default=2)
    train_parser.add_argument("--negative-samples", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=0.025)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--seed", type=int, default=1022)
    train_parser.add_argument("--limit", type=int)
    train_parser.set_defaults(func=train)

    similar_parser = subparsers.add_parser("similar")
    similar_parser.add_argument("--model", type=Path, required=True)
    similar_parser.add_argument(
        "--model-type",
        choices=["cooccurrence", "ngram", "sgns"],
        default="cooccurrence",
    )
    similar_parser.add_argument("--word", required=True)
    similar_parser.add_argument("--output", type=Path, default=Path("outputs/similar_words.csv"))
    similar_parser.add_argument("--topn", type=int, default=10)
    similar_parser.set_defaults(func=similar)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--model", type=Path, required=True)
    plot_parser.add_argument(
        "--model-type",
        choices=["cooccurrence", "ngram", "sgns"],
        default="cooccurrence",
    )
    plot_parser.add_argument("--output", type=Path, default=Path("outputs/tsne.png"))
    plot_parser.add_argument("--max-words", type=int, default=300)
    plot_parser.add_argument("--entities", action="store_true")
    plot_parser.set_defaults(func=plot)

    bpe_train_parser = subparsers.add_parser("bpe-train")
    bpe_train_parser.add_argument("--input", type=Path, required=True)
    bpe_train_parser.add_argument("--output", type=Path, required=True)
    bpe_train_parser.add_argument("--backend", choices=["simple", "hf"], default="simple")
    bpe_train_parser.add_argument("--num-merges", type=int, default=200)
    bpe_train_parser.add_argument("--vocab-size", type=int, default=30000)
    bpe_train_parser.add_argument("--min-frequency", type=int, default=2)
    bpe_train_parser.add_argument("--limit", type=int)
    bpe_train_parser.set_defaults(func=bpe_train)

    bpe_tokenize_parser = subparsers.add_parser("bpe-tokenize")
    bpe_tokenize_parser.add_argument("--model", type=Path, required=True)
    bpe_tokenize_parser.add_argument("--input", type=Path, required=True)
    bpe_tokenize_parser.add_argument("--output", type=Path, required=True)
    bpe_tokenize_parser.add_argument("--backend", choices=["simple", "hf"], default="simple")
    bpe_tokenize_parser.add_argument("--limit", type=int)
    bpe_tokenize_parser.set_defaults(func=bpe_tokenize)

    word2vec_parser = subparsers.add_parser("train-word2vec")
    word2vec_parser.add_argument("--input", type=Path, required=True)
    word2vec_parser.add_argument("--output", type=Path, required=True)
    word2vec_parser.add_argument("--vector-size", type=int, default=256)
    word2vec_parser.add_argument("--window-size", type=int, default=5)
    word2vec_parser.add_argument("--negative-samples", type=int, default=10)
    word2vec_parser.add_argument("--min-count", type=int, default=2)
    word2vec_parser.add_argument("--workers", type=int, default=4)
    word2vec_parser.add_argument("--epochs", type=int, default=10)
    word2vec_parser.add_argument("--limit", type=int)
    word2vec_parser.set_defaults(func=train_word2vec)

    mlm_parser = subparsers.add_parser("train-mlm-lora")
    mlm_parser.add_argument("--input", type=Path, required=True)
    mlm_parser.add_argument("--output-dir", type=Path, required=True)
    mlm_parser.add_argument("--model-name", default="bert-base-uncased")
    mlm_parser.add_argument("--max-length", type=int, default=256)
    mlm_parser.add_argument("--max-lines", type=int)
    mlm_parser.add_argument("--epochs", type=float, default=1.0)
    mlm_parser.add_argument("--batch-size", type=int, default=8)
    mlm_parser.add_argument("--learning-rate", type=float, default=5e-5)
    mlm_parser.add_argument("--lora-rank", type=int, default=4)
    mlm_parser.add_argument("--lora-alpha", type=int, default=32)
    mlm_parser.add_argument("--lora-dropout", type=float, default=0.1)
    mlm_parser.set_defaults(func=train_mlm)

    predict_parser = subparsers.add_parser("predict-next")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--context", required=True)
    predict_parser.add_argument("--topn", type=int, default=10)
    predict_parser.set_defaults(func=predict_next)

    inventory_parser = subparsers.add_parser("entity-inventory")
    inventory_parser.add_argument("--model", type=Path, required=True)
    inventory_parser.add_argument(
        "--model-type",
        choices=["cooccurrence", "ngram", "sgns"],
        default="cooccurrence",
    )
    inventory_parser.add_argument("--output", type=Path, default=Path("outputs/entities.csv"))
    inventory_parser.set_defaults(func=entity_inventory)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
