"""Questionnaire field definitions derived from the survey CSV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.survey_options import MULTI_OPTIONS_UI

FieldType = Literal["choice", "multi", "number", "text"]

RATING_1_10: tuple[str, ...] = tuple(str(i) for i in range(1, 11))


@dataclass(frozen=True)
class QuestionField:
    key: str
    label: str
    field_type: FieldType
    options: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    placeholder: str | None = None


# Column keys match persisted model feature order (see scripts/train_model.py).
QUESTION_FIELDS: tuple[QuestionField, ...] = (
    QuestionField(
        key="q1",
        label="Стабильность или новшество?",
        field_type="choice",
        options=("Стабильность", "Новшество"),
    ),
    QuestionField(
        key="q2",
        label="Часто ли вы фотографируете?",
        field_type="choice",
        options=("Каждый день", "Раз в неделю", "Раз в месяц"),
    ),
    QuestionField(
        key="q3",
        label="Ведёте ли вы блог?",
        field_type="choice",
        options=("Да", "Нет"),
    ),
    QuestionField(
        key="q4",
        label="За что вы готовы переплатить, покупая смартфон?",
        field_type="multi",
        options=MULTI_OPTIONS_UI,
    ),
    QuestionField(
        key="q5",
        label="Статус или технологичность?",
        field_type="choice",
        options=("Статус", "Технологичность"),
    ),
    QuestionField(
        key="q6",
        label='VPN или «запрет»?',
        field_type="choice",
        options=("VPN", "Запрет"),
    ),
    QuestionField(
        key="q7",
        label="Сколько вам лет?",
        field_type="number",
        min_value=5,
        max_value=99,
        step=1,
        placeholder="Например, 22",
    ),
    QuestionField(
        key="q8",
        label="Сколько в среднем вы готовы потратить на новый телефон? (₽)",
        field_type="number",
        min_value=0,
        max_value=500_000,
        step=1000,
        placeholder="Например, 70000",
    ),
    QuestionField(
        key="q9",
        label="Насколько вам важна возможность кастомизации интерфейса?",
        field_type="choice",
        options=RATING_1_10,
    ),
    QuestionField(
        key="q10",
        label="Можете ли вы поставить рингтон из файла?",
        field_type="choice",
        options=("Да", "Нет", "Не знаю"),
    ),
    QuestionField(
        key="q11",
        label="Как вы относитесь к перепродаже смартфона спустя 2–3 года?",
        field_type="choice",
        options=("Хорошо", "Плохо"),
    ),
    QuestionField(
        key="q12",
        label="Насколько вам важна возможность настройки системы под себя?",
        field_type="choice",
        options=("Да", "Нет", "Не знаю"),
    ),
    QuestionField(
        key="q13",
        label="Как часто вы меняете телефон? (в годах)",
        field_type="number",
        min_value=0.5,
        max_value=15,
        step=0.5,
        placeholder="Например, 3",
    ),
    QuestionField(
        key="q14",
        label="Насколько важна актуальность и обслуживание старых версий ОС?",
        field_type="choice",
        options=("Важна", "Не важна"),
    ),
    QuestionField(
        key="q15",
        label="Ваша сфера деятельности",
        field_type="text",
        placeholder="Например, IT, дизайн, студент",
    ),
    QuestionField(
        key="q16",
        label="Какого вы пола?",
        field_type="choice",
        options=("М", "Ж"),
    ),
    QuestionField(
        key="q17",
        label="Вы ведёте Instagram?",
        field_type="choice",
        options=("Да", "Нет"),
    ),
    QuestionField(
        key="q18",
        label="Сколько раз в день вы примерно заряжаете телефон?",
        field_type="number",
        min_value=0,
        max_value=10,
        step=0.5,
        placeholder="Например, 2",
    ),
    QuestionField(
        key="q19",
        label="Что бы вы выбрали: экономить и откладывать или порадовать себя?",
        field_type="choice",
        options=("Экономить и откладывать", "Порадовать себя"),
    ),
    QuestionField(
        key="q20",
        label="Ваша гарнитура от одного бренда или от разных?",
        field_type="choice",
        options=("От одного бренда", "От разных брендов"),
    ),
    QuestionField(
        key="q21",
        label="Как часто вы кушаете яблоки? (от 1 до 10)",
        field_type="choice",
        options=RATING_1_10,
    ),
)

FIELD_BY_KEY = {f.key: f for f in QUESTION_FIELDS}
