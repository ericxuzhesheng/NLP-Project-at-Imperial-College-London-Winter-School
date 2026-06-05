"""Masked language model fine-tuning utilities."""

from __future__ import annotations

from pathlib import Path


def train_mlm_lora(
    input_path: str | Path,
    output_dir: str | Path,
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    max_lines: int | None = None,
    epochs: float = 1.0,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    lora_rank: int = 4,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> None:
    """Fine-tune a masked language model with LoRA adapters."""

    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install transformers, datasets, and peft to train MLM-LoRA.") from exc

    texts = _read_texts(input_path, max_lines=max_lines)
    if not texts:
        raise ValueError("No non-empty training lines found.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, config)

    dataset = Dataset.from_dict({"text": texts}).train_test_split(test_size=0.1)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    output_dir = Path(output_dir)
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def _read_texts(path: str | Path, max_lines: int | None = None) -> list[str]:
    texts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_lines is not None and index >= max_lines:
                break
            text = line.strip()
            if text:
                texts.append(text)
    return texts
