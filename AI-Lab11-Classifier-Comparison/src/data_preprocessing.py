"""Data preprocessing for Lab 11.

Implements the Lab steps:
- Split into X and y
- Convert object columns to int codes (factorize)
- Train/test split (shuffle=False is used in train phase)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    # Use a robust parser for Kaggle-style exports.
    # If any rows have extra commas due to missing fields, skip them.
    # Robust parser to handle Kaggle-style exports with occasional malformed lines.
    return pd.read_csv(
        data_path,
        engine="python",
        on_bad_lines="skip",
    )




def encode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Factorize object columns into integer codes."""
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        out[obj_cols] = out[obj_cols].apply(lambda s: pd.factorize(s)[0])
    return out


def load_and_prepare(data_path: Path, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = load_dataset(data_path)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    # Lab 11 step: splitting into x and y
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    # Make sure y uses target_col if dataset column order changes
    y = df[target_col]
    x = df.drop(columns=[target_col])

    # Convert object columns into Int columns
    x_encoded = encode_object_columns(x)

    # Ensure no missing values for models
    # (Lab 11 pdf does not explicitly cover null handling, so we fill safe defaults.)
    x_encoded = x_encoded.fillna(x_encoded.mode(numeric_only=True).iloc[0])

    return df, x_encoded, y

