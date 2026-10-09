"""Веб-сервер: отдаёт страницу опроса и принимает ответы для предсказания."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ml import PIPELINE_PATH, ModelNotReadyError, get_metrics, predict, validate_answers
from app.schema import QUESTION_FIELDS

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="python_k_next", description="Опрос + KNN: iPhone vs Android")
# /static отдаёт CSS и JavaScript, Jinja2 подставляет вопросы и метрики в HTML.
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def _model_ready() -> bool:
    """Наличие файла проверяем без загрузки модели в память."""
    return PIPELINE_PATH.exists()


@app.get("/", response_class=HTMLResponse)
async def survey_page(request: Request) -> HTMLResponse:
    """Главная страница: форма или подсказка о необходимости обучения."""
    model_ready = _model_ready()
    metrics = {}
    model_error = None
    if model_ready:
        try:
            metrics = get_metrics()
        except ModelNotReadyError as exc:
            model_error = str(exc)
    else:
        model_error = "Модель не обучена. Запустите python scripts/train_model.py"

    return templates.TemplateResponse(
        request=request,
        name="survey.html",
        context={
            "questions": QUESTION_FIELDS,
            "metrics": metrics,
            "model_ready": model_ready and model_error is None,
            "model_error": model_error,
        },
    )


@app.get("/api/metrics")
async def api_metrics() -> JSONResponse:
    """Позволяет получить сохранённые метрики отдельно от страницы."""
    if not _model_ready():
        return JSONResponse({"error": "model_not_ready"}, status_code=503)
    return JSONResponse(get_metrics())


@app.post("/api/predict")
async def api_predict(request: Request) -> JSONResponse:
    """JSON от браузера → проверка → модель → JSON с результатом."""
    if not _model_ready():
        return JSONResponse(
            {"error": "model_not_ready", "message": "Модель недоступна"},
            status_code=503,
        )
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            {"error": "invalid_json", "message": "Некорректный запрос"},
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "invalid_json", "message": "Запрос должен быть JSON-объектом"},
            status_code=400,
        )

    answers = body.get("answers", {})
    # 422 — ошибки ответов; 503 — модели нет; 500 — ошибка предсказания.
    errors = validate_answers(answers)
    if errors:
        return JSONResponse({"error": "validation", "messages": errors}, status_code=422)

    try:
        result = predict(answers)
    except ModelNotReadyError as exc:
        return JSONResponse({"error": "model_not_ready", "message": str(exc)}, status_code=503)
    except Exception:
        return JSONResponse(
            {
                "error": "prediction_failed",
                "message": "Не удалось выполнить предсказание. Попробуйте ещё раз.",
            },
            status_code=500,
        )

    return JSONResponse(result)


@app.post("/predict")
async def predict_form_legacy() -> JSONResponse:
    """Старый адрес оставлен, чтобы объяснять клиентам, какой API использовать."""
    return JSONResponse(
        {
            "error": "use_api",
            "message": "Отправьте ответы через /api/predict (JSON), не через HTML-форму.",
        },
        status_code=410,
    )
