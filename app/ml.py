from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schema import QUESTION_FIELDS
from app.survey_options import MULTI_OPTIONS_FULL

NUMERIC_CHOICE_KEYS = frozenset({"q9", "q21"})

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
PIPELINE_PATH = ARTIFACT_DIR / "knn_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

_pipeline = None
_metrics: dict[str, Any] | None = None


class ModelNotReadyError(RuntimeError):
    pass


def _ensure_loaded() -> None:
    global _pipeline, _metrics
    if _pipeline is None:
        if not PIPELINE_PATH.exists():
            raise ModelNotReadyError(
                "Модель не найдена. Запустите: python scripts/train_model.py"
            )
        _pipeline = joblib.load(PIPELINE_PATH)
    if _metrics is None:
        if METRICS_PATH.exists():
            _metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        else:
            _metrics = {}


def get_metrics() -> dict[str, Any]:
    _ensure_loaded()
    return dict(_metrics or {})


def answers_to_frame(answers: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for field in QUESTION_FIELDS:
        key = field.key
        val = answers.get(key)
        if field.field_type == "multi":
            if isinstance(val, list):
                selected = {str(v).strip() for v in val}
            elif val:
                selected = {str(val).strip()}
            else:
                selected = set()
            for i, opt in enumerate(MULTI_OPTIONS_FULL):
                row[f"q4_{i}"] = 1 if opt in selected else 0
            continue
        elif key in NUMERIC_CHOICE_KEYS or field.field_type == "number":
            row[key] = float(val) if val is not None and val != "" else np.nan
        else:
            row[key] = str(val).strip() if val is not None else ""
    return pd.DataFrame([row])


def predict(answers: dict[str, Any]) -> dict[str, Any]:
    _ensure_loaded()
    frame = answers_to_frame(answers)
    model = _pipeline.named_steps["model"]
    preprocessor = _pipeline.named_steps["preprocess"]
    X = preprocessor.transform(frame)

    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    best_idx = int(np.argmax(proba))
    prediction = classes[best_idx]
    confidence = float(proba[best_idx])

    distances, indices = model.kneighbors(X, n_neighbors=min(model.n_neighbors, len(model._y)))
    def _neighbor_label(idx: int) -> str:
        raw = model._y[idx]
        classes = list(model.classes_)
        if raw in classes:
            return str(raw)
        try:
            return str(classes[int(raw)])
        except (TypeError, ValueError, IndexError):
            return str(raw)

    neighbor_labels = [_neighbor_label(int(i)) for i in indices[0]]
    neighbor_distances = [float(d) for d in distances[0]]

    label_ru = {"iPhone": "iPhone (Айфон)", "Android": "Android (Андроид)"}.get(
        prediction, prediction
    )

    return {
        "prediction": prediction,
        "prediction_label": label_ru,
        "confidence": confidence,
        "class_probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "neighbors": [
            {"label": lbl, "distance": dist}
            for lbl, dist in zip(neighbor_labels, neighbor_distances)
        ],
    }


def validate_answers(answers: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in QUESTION_FIELDS:
        val = answers.get(field.key)
        if field.field_type == "multi":
            selected = val if isinstance(val, list) else []
            if not selected:
                errors.append(f"Выберите хотя бы один вариант: {field.label}")
            continue
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"Заполните поле: {field.label}")
            continue
        if key in NUMERIC_CHOICE_KEYS or field.field_type == "number":
            try:
                num = float(val)
            except (TypeError, ValueError):
                errors.append(f"Некорректное число: {field.label}")
                continue
            if field.min_value is not None and num < field.min_value:
                errors.append(f"{field.label}: минимум {field.min_value}")
            if field.max_value is not None and num > field.max_value:
                errors.append(f"{field.label}: максимум {field.max_value}")
    return errors
