from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ml import ModelNotReadyError, get_metrics, predict, validate_answers
from app.schema import QUESTION_FIELDS

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="python_k_next", description="Опрос + KNN: iPhone vs Android")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def _model_ready() -> bool:
    return (APP_DIR.parent / "artifacts" / "knn_pipeline.joblib").exists()


@app.get("/", response_class=HTMLResponse)
async def survey_page(request: Request) -> HTMLResponse:
    metrics = {}
    model_error = None
    if _model_ready():
        try:
            metrics = get_metrics()
        except ModelNotReadyError as exc:
            model_error = str(exc)
    else:
        model_error = "Модель не обучена. Запустите python scripts/train_model.py"

    return templates.TemplateResponse(
        "survey.html",
        {
            "request": request,
            "questions": QUESTION_FIELDS,
            "metrics": metrics,
            "model_ready": _model_ready() and model_error is None,
            "model_error": model_error,
        },
    )


@app.get("/api/metrics")
async def api_metrics() -> JSONResponse:
    if not _model_ready():
        return JSONResponse({"error": "model_not_ready"}, status_code=503)
    return JSONResponse(get_metrics())


@app.post("/api/predict")
async def api_predict(request: Request) -> JSONResponse:
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

    answers = body.get("answers") or {}
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
    """Legacy HTML POST — use JSON /api/predict instead."""
    return JSONResponse(
        {
            "error": "use_api",
            "message": "Отправьте ответы через /api/predict (JSON), не через HTML-форму.",
        },
        status_code=410,
    )
