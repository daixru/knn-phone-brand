"""Проверяем сохранение поведения обучения и путь ответа от API до KNN."""

import asyncio
import json
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app import main, ml
from app.schema import QUESTION_FIELDS
from app.survey_options import MULTI_KEYS, MULTI_OPTIONS_FULL
from scripts import train_model


async def request_app(method, path, body=b""):
    """Отправляем запрос прямо в ASGI-приложение, без сети и HTTP-клиента."""
    messages = []
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path,
        "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234), "server": ("test", 80),
    }
    await main.app(scope, receive, send)
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    content = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return status, content


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features, cls.targets = train_model.load_frame()
        X_train, cls.X_test, y_train, cls.y_test = train_test_split(
            cls.features, cls.targets, test_size=train_model.TEST_SIZE,
            random_state=train_model.RANDOM_STATE, stratify=cls.targets,
        )
        cls.pipeline = train_model.build_pipeline()
        cls.pipeline.fit(X_train, y_train)
        # Восстанавливаем ответы первого участника из таблицы признаков.
        first = cls.features.iloc[0]
        cls.answers = {field.key: first[field.key] for field in QUESTION_FIELDS if field.key != "q4"}
        cls.answers["q4"] = [option for key, option in zip(MULTI_KEYS, MULTI_OPTIONS_FULL) if first[key] == 1]

    def setUp(self):
        # Проверки не зависят от наличия сохранённой модели и не меняют её файл.
        self.model_patch = patch.object(ml, "_pipeline", self.pipeline)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def api_request(self, method, path, body=b""):
        return asyncio.run(request_app(method, path, body))

    def test_training_quality_is_preserved(self):
        predictions = self.pipeline.predict(self.X_test)
        self.assertAlmostEqual(accuracy_score(self.y_test, predictions), 7 / 9)
        self.assertAlmostEqual(f1_score(self.y_test, predictions, average="macro"), 0.75)

    def test_csv_and_form_produce_same_features(self):
        frame = ml.answers_to_frame(self.answers)
        self.assertEqual(list(frame.columns), list(self.features.columns))
        np.testing.assert_allclose(
            self.pipeline.named_steps["preprocess"].transform(frame),
            self.pipeline.named_steps["preprocess"].transform(self.features.iloc[[0]]),
        )
        self.assertEqual(ml.validate_answers(self.answers), [])

    def test_prediction_and_neighbors(self):
        result = ml.predict(self.answers)
        frame = ml.answers_to_frame(self.answers)
        self.assertEqual(result["prediction"], self.pipeline.predict(frame)[0])
        self.assertAlmostEqual(sum(result["class_probabilities"].values()), 1)
        self.assertEqual(len(result["neighbors"]), 5)
        distances = [neighbor["distance"] for neighbor in result["neighbors"]]
        self.assertEqual(distances, sorted(distances))
        self.assertTrue(all(neighbor["label"] in ("iPhone", "Android") for neighbor in result["neighbors"]))

    def test_unknown_profession_is_supported(self):
        answers = dict(self.answers, q15="Ранее неизвестная профессия")
        self.assertIn(ml.predict(answers)["prediction"], ("iPhone", "Android"))

    def test_validation_reports_missing_and_invalid_numbers(self):
        self.assertEqual(len(ml.validate_answers({})), 21)
        for value in ("не число", "nan", "inf", -1, 500001):
            with self.subTest(value=value):
                self.assertTrue(ml.validate_answers(dict(self.answers, q8=value)))
        self.assertTrue(ml.validate_answers(dict(self.answers, q4=[])))
        self.assertTrue(ml.validate_answers(["не объект"]))

    def test_predict_api(self):
        payload = json.dumps({"answers": self.answers}, ensure_ascii=False).encode()
        with patch.object(main, "_model_ready", return_value=True):
            status, body = self.api_request("POST", "/api/predict", payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), ml.predict(self.answers))

    def test_api_invalid_requests(self):
        with patch.object(main, "_model_ready", return_value=True):
            for payload, expected in ((b"{", 400), (b"[]", 400), (b"null", 400),
                                      (b"{}", 422), (b'{"answers": []}', 422)):
                with self.subTest(payload=payload):
                    status, body = self.api_request("POST", "/api/predict", payload)
                    self.assertEqual(status, expected)
                    self.assertIn("error", json.loads(body))

    def test_page_metrics_and_missing_model(self):
        metrics = {
            "n_samples": 41, "n_train": 32, "n_test": 9,
            "accuracy": 7 / 9, "f1_macro": 0.75, "k_neighbors": 5,
            "question_influence": [{
                "label": "Тестовый вопрос", "association": 0.5,
                "knn_share": 0.2, "relative": 1.0, "knn_relative": 1.0,
            }],
        }
        with patch.object(main, "_model_ready", return_value=True), patch.object(main, "get_metrics", return_value=metrics):
            status, body = self.api_request("GET", "/")
            self.assertEqual(status, 200)
            html = body.decode()
            self.assertEqual(html.count('class="question-item"'), 21)
            self.assertIn('type="submit"', html)
            self.assertIn("Тестовый вопрос", html)
            status, body = self.api_request("GET", "/api/metrics")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), metrics)
        with patch.object(main, "_model_ready", return_value=False):
            for method, path in (("POST", "/api/predict"), ("GET", "/api/metrics")):
                status, _ = self.api_request(method, path)
                self.assertEqual(status, 503)
            status, body = self.api_request("GET", "/")
            self.assertEqual(status, 200)
            self.assertIn("python scripts/train_model.py", body.decode())
            self.assertNotIn('id="survey-form"', body.decode())

    def test_static_files_and_legacy_route(self):
        for path in ("/static/app.js", "/static/styles.css"):
            status, content = self.api_request("GET", path)
            self.assertEqual(status, 200)
            self.assertTrue(content)
        status, body = self.api_request("POST", "/predict")
        self.assertEqual(status, 410)
        self.assertEqual(json.loads(body)["error"], "use_api")


if __name__ == "__main__":
    unittest.main()
