"""Работа готовой модели: проверка ответов → таблица признаков → результат KNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schema import NUMERIC_KEYS, QUESTION_FIELDS
from app.survey_options import MULTI_KEYS, MULTI_OPTIONS_FULL

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
PIPELINE_PATH = ARTIFACT_DIR / "knn_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

# Загружаем модель один раз при первом обращении, затем используем из памяти.
_pipeline = None


class ModelNotReadyError(RuntimeError):
    """Понятная ошибка для случая, когда модель ещё не обучена."""


def _ensure_loaded() -> None:
    """Файл содержит и обработку признаков, и обученный классификатор."""
    global _pipeline
    if _pipeline is None:
        if not PIPELINE_PATH.exists():
            raise ModelNotReadyError(
                "Модель не найдена. Запустите: python scripts/train_model.py"
            )
        _pipeline = joblib.load(PIPELINE_PATH)


def get_metrics() -> dict[str, Any]:
    """Метрики читает сайт; их вычисление происходит в скрипте обучения."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    _ensure_loaded()
    return {}


def answers_to_frame(answers: dict[str, Any]) -> pd.DataFrame:
    """Один пользователь = одна строка, столбцы совпадают с обучающей таблицей."""
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
            for column, option in zip(MULTI_KEYS, MULTI_OPTIONS_FULL):
                row[column] = int(option in selected)
        elif key in NUMERIC_KEYS:
            row[key] = float(val) if val is not None and val != "" else np.nan
        else:
            row[key] = str(val).strip() if val is not None else ""
    return pd.DataFrame([row])


def predict(answers: dict[str, Any]) -> dict[str, Any]:
    """Возвращаем победивший класс, его долю голосов и ближайших соседей."""
    _ensure_loaded()
    frame = answers_to_frame(answers)
    model = _pipeline.named_steps["model"]
    preprocessor = _pipeline.named_steps["preprocess"]
    # Используем уже обученные кодировщик и масштабы; повторный fit здесь нельзя.
    features = preprocessor.transform(frame)

    # При weights="distance" это доли голосов с учётом расстояния,
    # а не гарантия того, какой телефон выберет человек.
    proba = model.predict_proba(features)[0]
    classes = list(model.classes_)
    best_idx = int(np.argmax(proba))
    prediction = classes[best_idx]
    confidence = float(proba[best_idx])

    # Внутри sklearn классы соседей хранятся как индексы: 0/1, а не названия.
    # _y — внутреннее поле sklearn; используем его только для показа соседей.
    # Сам прогноз и вероятности вычисляются публичным методом predict_proba.
    distances, indices = model.kneighbors(features)
    neighbor_labels = [str(classes[int(model._y[index])]) for index in indices[0]]
    neighbor_distances = [float(distance) for distance in distances[0]]

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
    """Проверка на сервере нужна даже при наличии ограничений в HTML."""
    if not isinstance(answers, dict):
        return ["Ответы должны быть объектом с ключами q1…q21"]
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
        if field.key in NUMERIC_KEYS:
            try:
                num = float(val)
            except (TypeError, ValueError):
                errors.append(f"Некорректное число: {field.label}")
                continue
            if not np.isfinite(num):
                errors.append(f"Некорректное число: {field.label}")
                continue
            if field.min_value is not None and num < field.min_value:
                errors.append(f"{field.label}: минимум {field.min_value}")
            if field.max_value is not None and num > field.max_value:
                errors.append(f"{field.label}: максимум {field.max_value}")
    return errors
