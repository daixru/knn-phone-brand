# python_k_next

Веб-опрос на русском языке и классификатор **k-ближайших соседей**, который по ответам предсказывает **iPhone** или **Android**. Обучение и поля формы основаны на файле `data/iphone-android-survey.csv`.

## Требования

- Python 3.11+

## Установка

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Обучение модели

```bash
python scripts/train_model.py
```

Скрипт делит данные на обучение/тест (75/25), обучает KNN, выводит метрики в консоль и сохраняет:

- `artifacts/knn_pipeline.joblib` — пайплайн предобработки + модель
- `artifacts/metrics.json` — точность, F1, матрица ошибок

## Запуск веб-приложения

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8742 --reload
```

Откройте в браузере: **http://127.0.0.1:8742**

## Структура

| Путь | Назначение |
|------|------------|
| `app/main.py` | FastAPI, маршруты и шаблоны |
| `app/ml.py` | Загрузка модели и предсказание |
| `app/schema.py` | Поля опроса (как в CSV) |
| `scripts/train_model.py` | Обучение KNN |
| `data/iphone-android-survey.csv` | Исходные ответы |
| `app/static/` | Стили и клиентский JS |
| `app/templates/` | HTML (Jinja2) |

Без базы данных, авторизации и внешних сервисов.
