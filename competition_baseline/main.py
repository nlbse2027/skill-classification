#!/usr/bin/env python3
"""Run the extracted SkillScope Random Forest baseline.

The model functions in this file are copied from the active Random Forest path
in ``src/classifier.py`` and ``main.py``.  In particular, this intentionally
preserves the original TF-IDF, MLSMOTE, duplicate-concatenation, and Random
Forest behavior.  See ``RF_BASELINE.md`` for the resulting limitations.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import string
from pathlib import Path

import emoji
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer


COMPETITION_TABLE = "nlbse_tool_competition_data_by_issue"
TEXT_COLUMNS = ["issue text", "issue description"]
COMPETITION_METADATA_COLUMNS = [
    "Repo Name",
    "PR #",
    "issue text",
    "issue description",
    "created_at",
    "closed_at",
    "userlogin",
    "author_name",
    "most_recent_commit",
]
ART_METADATA_COLUMNS = [
    "PR #",
    "Pull Request",
    "issue text",
    "issue description",
    "created_at",
    "closed_at",
    "userlogin",
    "author_name",
    "most_recent_commit",
    "filename",
    "file_commit",
    "api",
    "function_name",
    "api_domain",
    "subdomain",
]


def clean_text(text):
    """Copy of ``src.classifier.clean_text`` used during RF training."""
    cleaned_count = 0
    original_count = 0
    if not isinstance(text, str):
        original_count += 1
        return text

    text = text.replace('"', "")
    text = re.sub(r"DevTools.*?\(automated\)", "", text)
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [word for word in words if len(word) <= 20]
    cleaned_text = " ".join(words)

    cleaned_count += 1
    return cleaned_text


def extract_text_features(data):
    """Copy of the active SkillScope RF text-feature extraction."""
    X_text = data[["issue text", "issue description"]]
    X_text["combined_text"] = (
        X_text["issue text"] + " " + X_text["issue description"]
    )
    tfidf = TfidfVectorizer(max_features=1000)
    X_text_features = tfidf.fit_transform(X_text["combined_text"]).toarray()
    return X_text_features, tfidf


def transform_labels(data):
    """Copy of the active SkillScope RF label transformation."""
    y = data.drop(columns=["issue text", "issue description"])
    y = y.apply(lambda x: x.index[x == 1].tolist(), axis=1)
    mlb = MultiLabelBinarizer()
    y_transformed = mlb.fit_transform(y)
    y_df = pd.DataFrame(y_transformed, columns=mlb.classes_)
    return y_df, mlb


def create_combined_features(x_text_features):
    """Copy of the active SkillScope RF feature-frame construction."""
    X_combined = pd.DataFrame(
        x_text_features, columns=list(range(x_text_features.shape[1]))
    )
    return X_combined


def perform_mlsmote(X, y, n_sample):
    """Copy of the active SkillScope MLSMOTE implementation.

    ``n_sample`` is intentionally retained even though the implementation does
    not use it.
    """

    def nearest_neighbour(X):
        nbs = NearestNeighbors(
            n_neighbors=3, metric="euclidean", algorithm="kd_tree"
        ).fit(X)
        _, indices = nbs.kneighbors(X)
        return indices

    class_distribution = y.sum(axis=0)
    max_class_count = class_distribution.max()
    samples_needed = (max_class_count - class_distribution).astype(int)

    indices2 = nearest_neighbour(X.values)
    n = len(indices2)
    new_X = []
    target = []

    for class_idx, samples in samples_needed.items():
        if samples > 0:
            for _ in range(samples):
                reference = random.randint(0, n - 1)
                neighbour = random.choice(indices2[reference, 1:])
                all_point = indices2[reference]
                nn_df = y[y.index.isin(all_point)]
                ser = nn_df.sum(axis=0, skipna=True)
                new_target = np.array([1 if val > 2 else 0 for val in ser])
                ratio = random.random()
                gap = X.loc[reference, :] - X.loc[neighbour, :]
                new_sample = np.array(X.loc[reference, :] + ratio * gap)
                new_X.append(new_sample)
                target.append(new_target)

    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)

    X_combined = pd.concat([X, new_X], axis=0)
    y_combined = pd.concat([y, target], axis=0)

    return X_combined, y_combined


def train_random_forest(x_train, y_train):
    """Copy of the second, and therefore active, SkillScope RF definition."""
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)
    return clf


def load_dataset(path: Path, table: str) -> pd.DataFrame:
    """Load the official SQLite artifact or an equivalent CSV file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if table not in tables:
            available = ", ".join(sorted(tables)) or "(none)"
            raise ValueError(
                f"Table {table!r} was not found. Available tables: {available}"
            )
        escaped_table = table.replace('"', '""')
        return pd.read_sql_query(f'SELECT * FROM "{escaped_table}"', connection)


def find_label_columns(data: pd.DataFrame, label_start_column: str | None) -> list[str]:
    """Locate labels without silently treating metadata as target values."""
    columns = list(data.columns)

    if label_start_column is not None:
        if label_start_column not in columns:
            raise ValueError(f"Label-start column {label_start_column!r} was not found")
        return columns[columns.index(label_start_column) :]

    for metadata in (COMPETITION_METADATA_COLUMNS, ART_METADATA_COLUMNS):
        if columns[: len(metadata)] == metadata:
            return columns[len(metadata) :]

    raise ValueError(
        "Unrecognized dataset columns. Pass --label-start-column with the first "
        "label column."
    )


def prepare_dataset(
    data: pd.DataFrame, label_start_column: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the data preparation used before the original RF functions."""
    missing_text = [column for column in TEXT_COLUMNS if column not in data.columns]
    if missing_text:
        raise ValueError(f"Missing text columns: {', '.join(missing_text)}")

    label_columns = find_label_columns(data, label_start_column)
    if not label_columns:
        raise ValueError("No label columns were found")

    prepared = data[TEXT_COLUMNS + label_columns].copy()
    prepared[label_columns] = prepared[label_columns].map(
        lambda x: 1 if x > 0 else 0
    )
    prepared["issue text"] = prepared["issue text"].apply(clean_text)
    prepared["issue description"] = prepared["issue description"].apply(clean_text)
    prepared = prepared.dropna()
    return prepared, label_columns


def run_baseline(
    data: pd.DataFrame,
    label_start_column: str | None,
    test_size: float,
    split_seed: int,
) -> dict[str, float | int]:
    """Run the faithful model path and evaluate an augmented 80/20 holdout."""
    prepared, source_label_columns = prepare_dataset(data, label_start_column)

    print(f"Rows after preparation: {len(prepared)}")
    print(f"Source label columns: {len(source_label_columns)}")
    print("Extracting TF-IDF features from the complete dataset...")
    x_text_features, _ = extract_text_features(prepared)
    y_df, _ = transform_labels(prepared)
    x_combined = create_combined_features(x_text_features)

    class_distribution = y_df.sum(axis=0)
    synthetic_count = int(
        (class_distribution.max() - class_distribution).astype(int).sum()
    )
    print(f"Transformed label columns: {len(y_df.columns)}")
    print(f"MLSMOTE synthetic rows requested by the original code: {synthetic_count}")
    print("Balancing classes...")
    x_augmented, y_augmented = perform_mlsmote(
        x_combined, y_df, n_sample=500
    )

    # This second concatenation is intentional: it is the active SkillScope
    # training path, even though perform_mlsmote already returned the originals.
    x_combined = pd.concat([x_combined, x_augmented], axis=0)
    y_combined = pd.concat([y_df, y_augmented], axis=0)

    x_train, x_test, y_train, y_test = train_test_split(
        x_combined,
        y_combined,
        test_size=test_size,
        random_state=split_seed,
    )

    print(f"Augmented rows: {len(x_combined)}")
    print(f"Training rows: {len(x_train)}; test rows: {len(x_test)}")
    print("Training Random Forest...")
    classifier = train_random_forest(x_train, y_train)
    predictions = classifier.predict(x_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="micro"
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "labels": len(y_df.columns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the extracted SkillScope Random Forest baseline"
    )
    parser.add_argument("dataset", type=Path, help="Competition .db/.sqlite or CSV")
    parser.add_argument(
        "--table",
        default=COMPETITION_TABLE,
        help=f"SQLite table (default: {COMPETITION_TABLE})",
    )
    parser.add_argument(
        "--label-start-column",
        help="First label column; only needed for an unrecognized CSV schema",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Provisional augmented holdout fraction (default: 0.2)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the provisional holdout split (default: 42)",
    )
    parser.add_argument("--output", type=Path, help="Optional metrics JSON path")
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"Dataset does not exist: {args.dataset}")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    data = load_dataset(args.dataset, args.table)
    metrics = run_baseline(
        data,
        args.label_start_column,
        args.test_size,
        args.split_seed,
    )
    serialized = json.dumps(metrics, indent=2, sort_keys=True)
    print("\nMicro-averaged metrics:")
    print(serialized)

    if args.output is not None:
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Metrics written to {args.output}")


if __name__ == "__main__":
    main()
