"""Generate final test predictions using a fine-tuned BERTIN model.

Applies the threshold tuned on validation (see train.py) to the test set --
fixing the original notebook's bug, where the tuned threshold was computed
but never used: the final submission there called argmax on the logits,
which is equivalent to threshold 0.5, silently discarding the improvement
that had just been measured.

Run with::

    python -m src.predict --threshold 0.42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .evaluate import predict_with_threshold
from .preprocess import load_profner_datasets
from .train import MAX_LENGTH, MODEL_DIR

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "reports" / "predicciones_test.tsv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True,
                        help="Decision threshold tuned on validation (see train.py output)")
    args = parser.parse_args()

    _, _, test_df = load_profner_datasets()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    encodings = tokenizer(list(test_df["text"]), padding="max_length",
                          truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encodings).logits
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()

    predictions = predict_with_threshold(probs, args.threshold)
    submission = pd.DataFrame({"id": test_df["tweet_id"], "label": predictions})

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Saved {len(submission)} predictions -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
