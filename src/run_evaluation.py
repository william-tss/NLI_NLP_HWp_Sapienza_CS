import argparse
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate


def compute_metrics(eval_pred):
    """Computes accuracy for the NLI task."""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def softmax(x):
    """Computes softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def evaluate_and_save(trainer, dataset, tokenizer, split_name, model_name, output_dir):
    """
    Evaluates the model on a specific dataset split and saves the predictions to a CSV file.
    """
    print(f"\n--- Evaluating on {split_name} ---")

    # Tokenize the dataset
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Run prediction
    predictions = trainer.predict(tokenized_dataset)

    # Extract raw logits, compute probabilities and predicted classes
    logits = predictions.predictions
    probs = softmax(logits)
    pred_labels = np.argmax(logits, axis=-1)
    true_labels = predictions.label_ids

    # Print accuracy to console
    acc = np.mean(pred_labels == true_labels)
    print(f"Accuracy on {split_name}: {acc:.4f}")

    # Build a DataFrame for deep failure mode analysis
    df = pd.DataFrame({
        "premise": dataset["premise"],
        "hypothesis": dataset["hypothesis"],
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "prob_entailment": probs[:, 0],
        "prob_neutral": probs[:, 1],
        "prob_contradiction": probs[:, 2]
    })

    # Save to CSV in the results folder
    safe_model_name = model_name.replace("/", "_")
    csv_filename = f"predictions_{safe_model_name}_{split_name}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned NLI model on MNLI and ANLI datasets.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model checkpoint or Hugging Face Hub ID")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short name of the model for output files (e.g., distilbert, deberta-v3)")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory where CSV files will be saved")
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading tokenizer and model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3)

    # Initialize a DataCollator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create a dummy TrainingArguments (required by Trainer even for evaluation)
    # We set per_device_eval_batch_size to a higher value for faster inference
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=64,
        do_train=False,
        do_predict=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Load datasets
    print("\nLoading MNLI and ANLI datasets...")
    mnli = load_dataset("nyu-mll/multi_nli")
    anli = load_dataset("facebook/anli")

    # Filter out entries with invalid labels (-1) in MNLI validation sets just in case
    mnli_matched = mnli["validation_matched"].filter(lambda x: x["label"] in [0, 1, 2])
    mnli_mismatched = mnli["validation_mismatched"].filter(lambda x: x["label"] in [0, 1, 2])

    # Dictionary mapping standard names to their respective Hugging Face dataset objects
    evaluation_sets = {
        "mnli_matched": mnli_matched,
        "mnli_mismatched": mnli_mismatched,
        "anli_r1": anli["test_r1"],
        "anli_r2": anli["test_r2"],
        "anli_r3": anli["test_r3"]
    }

    # Evaluate sequentially on all 5 required sets
    for split_name, dataset_split in evaluation_sets.items():
        evaluate_and_save(
            trainer=trainer,
            dataset=dataset_split,
            tokenizer=tokenizer,
            split_name=split_name,
            model_name=args.model_name,
            output_dir=args.results_dir
        )

    print("\nEvaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()