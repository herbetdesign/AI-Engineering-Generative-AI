from fastapi.testclient import TestClient

from foodintent_ml import api


class FakePipeline:
    def predict(self, messages: list[str]) -> list[str]:
        return ["order_status"]

    def predict_proba(self, messages: list[str]) -> list[list[float]]:
        return [[0.7, 0.1, 0.1, 0.05, 0.05]]


def test_health_reports_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(api, "classifier", None)
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": False}


def test_classify_always_requires_human_review(monkeypatch) -> None:
    monkeypatch.setattr(api, "classifier", {"pipeline": FakePipeline()})
    response = TestClient(api.app).post(
        "/classify", json={"message": "Meu pedido está atrasado"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "intent": "order_status",
        "confidence": 0.7,
        "review_recommended": True,
    }
