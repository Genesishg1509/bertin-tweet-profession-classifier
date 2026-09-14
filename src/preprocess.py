"""Text cleaning and dataset loading for the ProfNER profession-mention task."""
from __future__ import annotations

import re

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "luisgasco/profner_classification_master"


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, mentions, hashtags, and non-Spanish characters."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-záéíóúüñç\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_profner_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test splits from Hugging Face and clean the text.

    Test labels are always -1 in the raw dataset -- they exist only so the
    Hugging Face Dataset format is happy. They must be predicted, never read.
    """
    dataset = load_dataset(DATASET_NAME)
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    for df in (train_df, val_df, test_df):
        df["text"] = df["text"].apply(clean_text)

    return train_df, val_df, test_df
