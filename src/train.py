import argparse
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Identificativo del modello su Hugging Face")
    parser.add_argument("--output_dir", type=str, default="./out", help="Cartella di destinazione per i checkpoint")
    parser.add_argument("--smoke_test", action="store_true", help="Esegue un test rapido su 2000 esempi")
    args = parser.parse_args()

    print(f"Caricamento tokenizer e modello: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2}
    )

    print("Caricamento dataset MNLI...")
    dataset = load_dataset("nyu-mll/multi_nli")

    # Filtriamo eventuali esempi con label difettose
    train_ds = dataset["train"].filter(lambda x: x["label"] in [0, 1, 2])
    val_ds = dataset["validation_matched"].filter(lambda x: x["label"] in [0, 1, 2])

    if args.smoke_test:
        print("SMOKE TEST: Seleziono 2000 esempi di train e 500 di validation.")
        train_ds = train_ds.select(range(2000))
        val_ds = val_ds.select(range(500))

    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=128
        )

    print("Tokenizzazione in corso...")
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_val = val_ds.map(preprocess_function, batched=True)

    # Inizializza il Data Collator passandogli il tokenizer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Iperparametri definiti nella guida
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        num_train_epochs=1 if args.smoke_test else 2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch" if args.smoke_test else "steps",
        eval_steps=2000 if not args.smoke_test else None,
        save_strategy="no" if args.smoke_test else "steps",
        save_steps=2000 if not args.smoke_test else None,
        seed=42,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        data_collator=data_collator,  # <--- USA IL DATA COLLATOR INVECE DI TOKENIZER
    )

    print("Inizio addestramento...")
    trainer.train()


if __name__ == "__main__":
    main()