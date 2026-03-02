"""API smoke tests."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.app.api import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("product_count", body)

    def test_demo_page_served(self) -> None:
        response = self.client.get("/demo/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))

    def test_recommend_endpoint(self) -> None:
        payload = {
            "query_text": "vegan iced espresso",
            "top_k": 3,
            "include_explanations": True,
            "explanation_top_n": 2,
        }
        response = self.client.post("/v1/recommend", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("constraints", body)
        self.assertIn("product_ids", body)
        self.assertIn("top_explanations", body)

    def test_config_endpoint(self) -> None:
        response = self.client.get("/v1/config")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("mode", body)
        self.assertIn("model_preset", body)
        self.assertIn("top_k_default", body)

    def test_batch_recommend_endpoint(self) -> None:
        payload = {
            "items": [
                {"query_id": "Q1", "query_text": "hot brewed coffee"},
                {"query_id": "Q2", "query_text": "vegan iced espresso"},
            ],
            "top_k": 2,
            "include_explanations": False,
            "explanation_top_n": 2,
        }
        response = self.client.post("/v1/recommend/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["results"]), 2)

    def test_batch_recommend_empty_items(self) -> None:
        payload = {
            "items": [],
            "top_k": 2,
            "include_explanations": False,
            "explanation_top_n": 2,
        }
        response = self.client.post("/v1/recommend/batch", json=payload)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
