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

from app.schema import QUESTION_FIELDS
from app.survey_options import MULTI_OPTIONS_FULL

DATA_PATH = ROOT / "data" / "iphone-android-survey.csv"
ARTIFACT_DIR = ROOT / "artifacts"

MULTI_OPTIONS = MULTI_OPTIONS_FULL
MULTI_KEYS = [f"q4_{i}" for i in range(len(MULTI_OPTIONS))]
NUMERIC_KEYS = frozenset({"q7", "q8", "q9", "q13", "q18", "q21"})
TEST_SIZE = 0.2

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


def _question_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for field in QUESTION_FIELDS:
        if field.key == "q4":
            groups["q4"] = [c for c in MULTI_KEYS if c in X.columns]
        elif field.key in X.columns:
            groups[field.key] = [field.key]
    return groups


def _cramers_v(feature: pd.Series, target: np.ndarray) -> float:
    """Bias-corrected Cramér's V so high-cardinality fields are not inflated."""
    table = pd.crosstab(feature.astype(str).fillna(""), pd.Series(target))
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n <= 1:
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    denom = np.where(expected == 0, 1.0, expected)
    chi2 = float(np.sum((observed - expected) ** 2 / denom))
    r, k = observed.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom_corr = min(k_corr - 1, r_corr - 1)
    if denom_corr <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom_corr))


def _numeric_association(feature: pd.Series, target: np.ndarray) -> float:
    y_bin = pd.Series((np.asarray(target) == "iPhone").astype(float), index=feature.index)
    r = pd.to_numeric(feature, errors="coerce").corr(y_bin)
    if r is None or not np.isfinite(r):
        return 0.0
    return float(abs(r))


def _question_association(X: pd.DataFrame, y: np.ndarray, key: str, cols: list[str]) -> float:
    if not cols:
        return 0.0
    if key == "q4":
        combo = X[cols].fillna(0).astype(int).astype(str).agg("".join, axis=1)
        return _cramers_v(combo, y)
    series = X[cols[0]]
    if key in NUMERIC_KEYS:
        return _numeric_association(series, y)
    return _cramers_v(series, y)


def _grouped_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: dict[str, list[str]],
    n_repeats: int = 40,
    random_state: int = 42,
) -> tuple[float, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(random_state)
    baseline = float(accuracy_score(y, pipeline.predict(X)))
    originals = {col: X[col].to_numpy(copy=True) for col in X.columns}
    out: dict[str, dict[str, float]] = {}
    for key, cols in groups.items():
        drops: list[float] = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            for col in cols:
                Xp[col] = originals[col][perm]
            acc = float(accuracy_score(y, pipeline.predict(Xp)))
            drops.append(baseline - acc)
        out[key] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return baseline, out


def question_influence(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> list[dict[str, object]]:
    groups = _question_groups(X)
    _, perm = _grouped_permutation_importance(pipeline, X, y, groups)
    labels = {field.key: field.label for field in QUESTION_FIELDS}

    rows: list[dict[str, object]] = []
    for key, cols in groups.items():
        association = _question_association(X, y, key, cols)
        perm_mean = float(perm[key]["mean"])
        rows.append(
            {
                "key": key,
                "label": labels.get(key, key),
                "association": association,
                "permutation_importance": perm_mean,
                "permutation_std": float(perm[key]["std"]),
            }
        )

    max_assoc = max((float(r["association"]) for r in rows), default=0.0)
    perm_clipped = [max(float(r["permutation_importance"]), 0.0) for r in rows]
    perm_sum = float(sum(perm_clipped))
    max_perm = max(perm_clipped, default=0.0)
    for row, perm_pos in zip(rows, perm_clipped):
        assoc = float(row["association"])
        row["relative"] = float(assoc / max_assoc) if max_assoc > 0 else 0.0
        row["knn_share"] = float(perm_pos / perm_sum) if perm_sum > 0 else 0.0
        row["knn_relative"] = float(perm_pos / max_perm) if max_perm > 0 else 0.0

    rows.sort(
        key=lambda r: (float(r["association"]), float(r["permutation_importance"])),
        reverse=True,
    )
    return rows


def main() -> int:
    X, y = load_frame()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True)
    influence = question_influence(pipeline, X, y)

    metrics = {
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_size": TEST_SIZE,
        "k_neighbors": 5,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "question_influence": influence,
        "question_influence_note": (
            "association — сила связи ответа с классом (Cramér's V с поправкой на число "
            "вариантов для категорий, |корреляция| для чисел). permutation_importance — "
            "падение точности KNN при перемешивании вопроса."
        ),
    }

    joblib.dump(pipeline, ARTIFACT_DIR / "knn_pipeline.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== KNN evaluation (hold-out 20%) ===")
    print(f"Samples: {metrics['n_samples']} (train {metrics['n_train']}, test {metrics['n_test']})")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 (macro): {metrics['f1_macro']:.3f}")
    print("Confusion matrix (rows=true, cols=pred):", labels)
    for row in cm:
        print(" ", row)
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=labels))
    print("=== Question influence (class membership) ===")
    print(f"{'assoc':>7}  {'knn':>7}  question")
    for item in influence:
        print(
            f"{float(item['association']):7.3f}  {float(item['permutation_importance']):7.3f}  "
            f"{item['label']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
