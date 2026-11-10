"""FastAPI endpoints for the experimental FoodIntent model."""

from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from foodintent_ml.train import MODEL_PATH

app = FastAPI(
    title="FoodIntent ML",
    description=(
        "Experimental classifier for Portuguese delivery messages. "
        "Every result requires human review."
    ),
    version="0.1.0",
)
classifier: Any | None = None


@app.on_event("startup")
def initialize_model() -> None:
    """Load a trained local model when one exists."""
    global classifier
    if Path(MODEL_PATH).is_file():
        classifier = joblib.load(MODEL_PATH)


class ClassifyRequest(BaseModel):
    """Classification request."""

    message: str = Field(min_length=2, max_length=1_000)


class ClassifyResponse(BaseModel):
    """Classification response."""

    intent: str
    confidence: float
    review_recommended: bool


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Return API state and local model availability."""
    return {"status": "ok", "model_ready": classifier is not None}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Predict an intent and keep human review enabled for every prediction."""
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo indisponível. Execute python -m foodintent_ml.train.",
        )

    pipeline = classifier["pipeline"]
    intent = str(pipeline.predict([request.message])[0])
    confidence = float(max(pipeline.predict_proba([request.message])[0]))
    return ClassifyResponse(
        intent=intent,
        confidence=confidence,
        review_recommended=True,
    )
