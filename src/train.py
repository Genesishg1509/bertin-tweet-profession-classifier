"""Fine-tune BERTIN RoBERTa on the ProfNER profession-mention task.

Run with::

    python -m src.train

Needs a GPU to finish in reasonable time (roughly 15 minutes on a T4 for 4
epochs over 2,786 examples). On CPU this will be very slow -- Google Colab
or any CUDA machine is recommended.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from .evaluate import evaluation_report, find_best_threshold, predict_with_threshold
from .preprocess import load_profner_datasets

MODEL_NAME = "bertin-project/bertin-roberta-base-spanish"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "bertin"
REPORTS = Path(__file__).resolve().parents[1] / "reports"
MAX_LENGTH = 256


def _tokenize(df, tokenizer):
    def _preprocess(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=MAX_LENGTH)
    return Dataset.from_pandas(df).map(_preprocess, batched=True)


def _compute_metrics(eval_pred):
    import evaluate as hf_evaluate
    accuracy = hf_evaluate.load("accuracy")
    f1 = hf_evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"],
    }


def main() -> dict:
    train_df, val_df, test_df = load_profner_datasets()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = _tokenize(train_df, tokenizer)
    val_ds = _tokenize(val_df, tokenizer)

    id2label = {0: "NO_MENCIONA_PROF", 1: "MENCIONA_PROF"}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir="training_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.1,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none",
        seed=52,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Tune the decision threshold on validation, then evaluate with it --
    # the original notebook computed this but never applied it. See
    # evaluate.py's docstring for the honesty caveat on what this number means.
    import torch
    trainer.compute_metrics = None
    pred = trainer.predict(val_ds)
    probs_val = torch.softmax(torch.tensor(pred.predictions), dim=1)[:, 1].numpy()
    y_true = pred.label_ids

    best_t, _ = find_best_threshold(y_true, probs_val)
    y_pred = predict_with_threshold(probs_val, best_t)
    report = evaluation_report(y_true, y_pred)
    report["threshold"] = best_t
    report["model"] = "BERTIN RoBERTa (fine-tuned)"

    print(f"{report['model']}  (threshold={best_t:.2f})")
    print(f"  F1:       {report['f1']:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "bertin_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    main()
