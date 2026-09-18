#!/usr/bin/env python3
"""Train KNN classifier on survey data and persist model + metrics."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.survey_options import MULTI_OPTIONS_FULL

DATA_PATH = ROOT / "data" / "iphone-android-survey.csv"
ARTIFACT_DIR = ROOT / "artifacts"

MULTI_OPTIONS = MULTI_OPTIONS_FULL
MULTI_KEYS = [f"q4_{i}" for i in range(len(MULTI_OPTIONS))]

TARGET_MAP = {
    "айфон": "iPhone",
    "iphone": "iPhone",
    "андроид": "Android",
    "android": "Android",
}


def normalize_target(value: str) -> str:
    key = str(value).strip().lower()
    if key in TARGET_MAP:
        return TARGET_MAP[key]
    raise ValueError(f"Unknown target label: {value!r}")


def parse_price(raw: str) -> float:
    s = str(raw).strip().replace(" ", "").replace("\u00a0", "")
    if not s:
        return np.nan
    # European-style thousands: 100.000 -> 100000
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_float(raw: str) -> float:
    s = str(raw).strip().replace(",", ".")
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def multi_vector(raw: str) -> list[int]:
    parts = {p.strip() for p in str(raw).split(";") if p.strip()}
    return [1 if opt in parts else 0 for opt in MULTI_OPTIONS]


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    feature_cols = list(df.columns[1:-1])
    rows: list[dict[str, object]] = []
    targets: list[str] = []

    for _, row in df.iterrows():
        targets.append(normalize_target(row.iloc[-1]))
        record: dict[str, object] = {}
        for idx, col in enumerate(feature_cols):
            key = f"q{idx + 1}"
            val = row[col]
            if key == "q4":
                for i, bit in enumerate(multi_vector(val)):
                    record[f"q4_{i}"] = bit
            elif key in ("q7", "q8", "q13", "q18"):
                record[key] = parse_price(val) if key == "q8" else parse_float(val)
            elif key in ("q9", "q21"):
                record[key] = parse_float(val)
            else:
                record[key] = str(val).strip() if pd.notna(val) else ""
        rows.append(record)

    X = pd.DataFrame(rows)
    y = np.array(targets)
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    cat_cols = ["q1", "q2", "q3", "q5", "q6", "q10", "q11", "q12", "q14", "q16", "q17", "q19", "q20"]
    num_cols = ["q7", "q8", "q9", "q13", "q18", "q21"]
    multi_cols = MULTI_KEYS
    text_cols = ["q15"]

    transformers = []
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    if text_cols:
        transformers.append(
            (
                "text",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                text_cols,
            )
        )
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if multi_cols:
        transformers.append(
            (
                "multi",
                Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0))]),
                multi_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    clf = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2)
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def main() -> int:
    X, y = load_frame()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True)

    metrics = {
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "k_neighbors": 5,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    joblib.dump(pipeline, ARTIFACT_DIR / "knn_pipeline.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== KNN evaluation (hold-out 25%) ===")
    print(f"Samples: {metrics['n_samples']} (train {metrics['n_train']}, test {metrics['n_test']})")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 (macro): {metrics['f1_macro']:.3f}")
    print("Confusion matrix (rows=true, cols=pred):", labels)
    for row in cm:
        print(" ", row)
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
