#!/usr/bin/env python3
"""Обучение: CSV → признаки → разделение данных → KNN → метрики и файлы."""

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
# Позволяет запускать файл командой python scripts/train_model.py.
sys.path.insert(0, str(ROOT))

from app.schema import CATEGORY_KEYS, NUMERIC_KEYS, QUESTION_FIELDS, TEXT_KEYS
from app.survey_options import MULTI_KEYS, MULTI_OPTIONS_FULL

DATA_PATH = ROOT / "data" / "iphone-android-survey.csv"
ARTIFACT_DIR = ROOT / "artifacts"

TEST_SIZE = 0.2
K_NEIGHBORS = 5
RANDOM_STATE = 42

TARGET_MAP = {
    "айфон": "iPhone",
    "iphone": "iPhone",
    "андроид": "Android",
    "android": "Android",
}


def normalize_target(value: str) -> str:
    """Приводим русское и английское написание к двум названиям классов."""
    key = str(value).strip().lower()
    if key in TARGET_MAP:
        return TARGET_MAP[key]
    raise ValueError(f"Неизвестный класс телефона: {value!r}")


def parse_price(raw: str) -> float:
    """Распознаём цены вроде 70 000 и 100.000; ошибочные значения станут пропусками."""
    s = str(raw).strip().replace(" ", "").replace("\u00a0", "")
    if not s:
        return np.nan
    # Точки между группами из трёх цифр считаем разделителями тысяч.
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_float(raw: str) -> float:
    """Читаем число с точкой или запятой; пропуски позже заполнит SimpleImputer."""
    s = str(raw).strip().replace(",", ".")
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def multi_vector(raw: str) -> list[int]:
    """Например, «Статус;Камера» → [1, 1, 0, 0, 0]."""
    parts = {p.strip() for p in str(raw).split(";") if p.strip()}
    return [1 if opt in parts else 0 for opt in MULTI_OPTIONS_FULL]


def load_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """X — ответы (признаки), y — известный телефон каждого участника."""
    df = pd.read_csv(DATA_PATH)
    # Первый столбец — время, последний — телефон. Они не входят в признаки.
    feature_cols = list(df.columns[1:-1])
    if len(feature_cols) != len(QUESTION_FIELDS):
        raise ValueError("Количество вопросов в CSV не совпадает с app/schema.py")
    rows: list[dict[str, object]] = []
    targets: list[str] = []

    for _, row in df.iterrows():
        targets.append(normalize_target(row.iloc[-1]))
        record: dict[str, object] = {}
        for field, col in zip(QUESTION_FIELDS, feature_cols):
            key = field.key
            val = row[col]
            if key == "q4":
                for i, bit in enumerate(multi_vector(val)):
                    record[f"q4_{i}"] = bit
            elif key in NUMERIC_KEYS:
                record[key] = parse_price(val) if key == "q8" else parse_float(val)
            else:
                record[key] = str(val).strip() if pd.notna(val) else ""
        rows.append(record)

    X = pd.DataFrame(rows)
    y = np.array(targets)
    return X, y


def build_pipeline() -> Pipeline:
    """Pipeline выполняет одну и ту же подготовку при обучении и предсказании."""
    # Категории нельзя кодировать числами 1, 2, 3: это создало бы ложный порядок.
    # One-hot создаёт отдельный столбец 0/1 для каждого варианта ответа.
    categories = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Сфера деятельности тоже кодируется как категория, а не анализируется как текст.
    # Новый вариант не вызывает ошибку: handle_unknown="ignore" даёт нули.
    text = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Пропуски заменяем медианой. Масштабирование не даёт цене в рублях
    # перекрыть возраст и оценки 1–10 при вычислении расстояния.
    numbers = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    multiple_choice = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
    ])
    # Порядок групп оставляем прежним для воспроизводимости расстояний.
    preprocessor = ColumnTransformer([
        ("cat", categories, list(CATEGORY_KEYS)),
        ("text", text, list(TEXT_KEYS)),
        ("num", numbers, list(NUMERIC_KEYS)),
        ("multi", multiple_choice, list(MULTI_KEYS)),
    ])
    # p=2 — обычное евклидово расстояние. Близкие соседи голосуют сильнее дальних.
    model = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS, weights="distance", metric="minkowski", p=2
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _question_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    # Все пять столбцов q4 перемешиваем вместе, как один ответ.
    groups: dict[str, list[str]] = {}
    for field in QUESTION_FIELDS:
        if field.key == "q4":
            groups["q4"] = [c for c in MULTI_KEYS if c in X.columns]
        elif field.key in X.columns:
            groups[field.key] = [field.key]
    return groups


def _cramers_v(feature: pd.Series, target: np.ndarray) -> float:
    """Связь категории с телефоном (0…1), с поправкой на число вариантов.

    Сравниваем реальные частоты с ожидаемыми при отсутствии связи.
    Поправка уменьшает завышение оценки у вопросов с множеством вариантов.
    """
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
    """Модуль корреляции числа с классом: iPhone = 1, Android = 0."""
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
    random_state: int = RANDOM_STATE,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Перемешиваем ответы на один вопрос и измеряем падение точности.

    Повторяем 40 раз: mean — среднее падение, std — разброс результатов.
    Отрицательное падение возможно: после перемешивания точность выросла.
    """
    rng = np.random.default_rng(random_state)
    baseline = float(accuracy_score(y, pipeline.predict(X)))
    originals = {col: X[col].to_numpy(copy=True) for col in X.columns}
    out: dict[str, dict[str, float]] = {}
    for key, cols in groups.items():
        drops: list[float] = []
        for _ in range(n_repeats):
            shuffled_answers = X.copy()
            shuffled_indices = rng.permutation(len(X))
            for col in cols:
                shuffled_answers[col] = originals[col][shuffled_indices]
            acc = float(accuracy_score(y, pipeline.predict(shuffled_answers)))
            drops.append(baseline - acc)
        out[key] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return baseline, out


def question_influence(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> list[dict[str, object]]:
    """Две разные оценки: связь с классом и влияние на точность KNN."""
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

    # Для полос на странице нормируем оценки. Отрицательное падение точности
    # сохраняем в метриках, но для долей и ширины полос считаем равным нулю.
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
    # 1. Читаем исходные ответы и известные классы телефонов.
    X, y = load_frame()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 80% для обучения, 20% для проверки. stratify сохраняет доли классов.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Подготовка признаков обучается только на тренировочных данных.
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Проверяем модель на ответах, которые она не видела при обучении.
    y_pred = pipeline.predict(X_test)
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True)
    # Влияние, как и прежде, считаем по всей выборке. Это описательная оценка,
    # а не независимая проверка: часть этих ответов модель уже видела.
    influence = question_influence(pipeline, X, y)

    metrics = {
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_size": TEST_SIZE,
        "k_neighbors": K_NEIGHBORS,
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

    # 5. Сохраняем подготовку и модель вместе: сайт ничего не обучает заново.
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
