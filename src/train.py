import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def compute_metrics(eval_pred):
    """Compute accuracy from the logits returned by the model."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float((predictions == labels).mean())}


def tokenize_dataset(dataset, tokenizer):
    """Tokenize MNLI pairs and expose the target with the standard `labels` name."""

    def preprocess(examples):
        encoded = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=128,
        )
        encoded["labels"] = examples["label"]
        return encoded

    return dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing premise-hypothesis pairs",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Transformer for three-way NLI on MNLI."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model identifier")
    parser.add_argument(
        "--output_dir", default="./out", help="Directory for checkpoints and the final model"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override the default learning rate (2e-5 for DistilBERT, 1e-5 for DeBERTa).",
    )
    parser.add_argument("--epochs", type=float, default=None, help="Override the number of epochs.")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Train on 2,000 examples and validate on 500 examples to test the full pipeline.",
    )
    parser.add_argument(
        "--overfit_test",
        action="store_true",
        help="Train and evaluate on the same 64 examples; this must reach very high accuracy.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in output_dir.",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable mixed-precision training even when CUDA is available.",
    )
    args = parser.parse_args()

    if args.smoke_test and args.overfit_test:
        parser.error("Use either --smoke_test or --overfit_test, not both.")

    set_seed(42)
    output_dir = Path(args.output_dir)
    final_model_dir = output_dir / "final_model"
    is_deberta = "deberta" in args.model.lower()
    learning_rate = args.learning_rate or (1e-5 if is_deberta else 2e-5)

    if args.overfit_test:
        train_size, validation_size, default_epochs, batch_size, warmup_ratio = 64, 64, 30, 8, 0.0
        print("OVERFIT TEST: train and validation are the same 64 MNLI examples.")
    elif args.smoke_test:
        train_size, validation_size, default_epochs, batch_size, warmup_ratio = 2_000, 500, 3, 16, 0.06
        print("SMOKE TEST: 2,000 training examples and 500 validation examples.")
    else:
        train_size, validation_size, default_epochs, batch_size, warmup_ratio = None, None, 2, 16, 0.06
        print("FULL TRAINING: full MNLI train split and matched validation split.")

    epochs = args.epochs or default_epochs
    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    print(f"CUDA available: {torch.cuda.is_available()} | fp16 enabled: {use_fp16}")
    print(f"Learning rate: {learning_rate} | Epochs: {epochs} | Batch size: {batch_size}")

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=LABEL_NAMES,
        label2id={name: index for index, name in LABEL_NAMES.items()},
    )

    print("Loading MNLI...")
    dataset = load_dataset("nyu-mll/multi_nli")
    train_ds = dataset["train"].filter(lambda example: example["label"] in LABEL_NAMES)
    validation_ds = dataset["validation_matched"].filter(
        lambda example: example["label"] in LABEL_NAMES
    )

    if args.overfit_test:
        train_ds = train_ds.select(range(train_size))
        validation_ds = train_ds
    elif args.smoke_test:
        train_ds = train_ds.select(range(train_size))
        validation_ds = validation_ds.select(range(validation_size))

    print(f"Train examples: {len(train_ds):,} | labels: {dict(sorted(Counter(train_ds['label']).items()))}")
    print(
        f"Validation examples: {len(validation_ds):,} | "
        f"labels: {dict(sorted(Counter(validation_ds['label']).items()))}"
    )

    tokenized_train = tokenize_dataset(train_ds, tokenizer)
    tokenized_validation = tokenize_dataset(validation_ds, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Both strategies are "epoch": this is required when loading the best checkpoint.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=25 if args.overfit_test else 100,
        report_to="none",
        seed=42,
        data_seed=42,
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    metrics = trainer.evaluate()
    print(f"Final validation accuracy: {metrics['eval_accuracy']:.4f}")

    # After load_best_model_at_end, trainer.model is the best validation checkpoint.
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"Best model and tokenizer saved to: {final_model_dir}")

    if args.overfit_test:
        print("Expected result: accuracy close to 1.00. A low result indicates a pipeline bug.")


if __name__ == "__main__":
    main()
