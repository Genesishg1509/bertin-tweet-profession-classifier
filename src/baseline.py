"""TF-IDF + Logistic Regression baseline.

BERTIN is a ~110M-parameter transformer that needs a GPU to fine-tune in
reasonable time. Before paying that cost, it's worth checking how far a
linear model on bag-of-words features gets -- trains in seconds, on a
laptop, no GPU. If it lands close to BERTIN's F1, that's a real finding:
the transformer's complexity may not be earning its cost for this task.

Run with::

    python -m src.baseline
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .evaluate import evaluation_report, find_best_threshold, predict_with_threshold
from .preprocess import load_profner_datasets

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> dict:
    train_df, val_df, test_df = load_profner_datasets()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, train_df["label"])

    probs_val = model.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_threshold(val_df["label"].values, probs_val)
    y_pred_val = predict_with_threshold(probs_val, best_t)

    report = evaluation_report(val_df["label"].values, y_pred_val)
    report["threshold"] = best_t
    report["model"] = "TF-IDF + Logistic Regression"

    print(f"{report['model']}  (threshold={best_t:.2f})")
    print(f"  F1:       {report['f1']:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Confusion matrix: {report['confusion_matrix']}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "baseline_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    main()
