"""Train candidates and evaluate on separate source and curated splits."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from foodintent_ml.dataset import TARGET_NAMES, load_curated, load_split

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "foodintent_pipeline.joblib"
REPORTS_PATH = ROOT / "reports"
CURATED_WEIGHT = 8.0


def build_candidates() -> dict[str, Pipeline]:
    """Create word-level TF-IDF baselines with two classical classifiers."""

    def vectorizer() -> TfidfVectorizer:
        return TfidfVectorizer(
            ngram_range=(1, 2), max_features=30_000, strip_accents="unicode"
        )

    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", vectorizer()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_500, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "multinomial_nb": Pipeline(
            [("tfidf", vectorizer()), ("classifier", MultinomialNB(alpha=0.5))]
        ),
    }


def evaluate(model: Pipeline, frame: pd.DataFrame) -> dict[str, object]:
    """Return metrics using only labels represented in this evaluation split."""
    labels = [label for label in TARGET_NAMES if label in set(frame["intent"])]
    predictions = model.predict(frame["text"])
    return {
        "rows": len(frame),
        "accuracy": accuracy_score(frame["intent"], predictions),
        "macro_f1": f1_score(
            frame["intent"],
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "classification_report": classification_report(
            frame["intent"],
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            frame["intent"], predictions, labels=labels
        ).tolist(),
    }


def main() -> None:
    """Select by curated validation F1 and report two held-out test sets."""
    source_train = load_split("train")
    source_validation = load_split("validation")
    source_test = load_split("test")
    curated_train = load_curated("train")
    curated_validation = load_curated("validation")
    curated_test = load_curated("test")

    training = pd.concat([source_train, curated_train], ignore_index=True)
    sample_weights = np.where(
        training["origin"].eq("project_authored_synthetic"),
        CURATED_WEIGHT,
        1.0,
    )
    if training.empty or curated_validation.empty or curated_test.empty:
        raise ValueError("A required training or curated split is empty.")

    candidates = build_candidates()
    validation_scores: dict[str, float] = {}
    for name, model in candidates.items():
        model.fit(
            training["text"],
            training["intent"],
            classifier__sample_weight=sample_weights,
        )
        prediction = model.predict(curated_validation["text"])
        validation_scores[name] = f1_score(
            curated_validation["intent"],
            prediction,
            labels=TARGET_NAMES,
            average="macro",
            zero_division=0,
        )

    selected_name = max(validation_scores, key=validation_scores.get)
    selected_model = candidates[selected_name]
    source_metrics = evaluate(selected_model, source_test)
    curated_metrics = evaluate(selected_model, curated_test)
    curated_predictions = selected_model.predict(curated_test["text"])
    probability_matrix = selected_model.predict_proba(curated_test["text"])
    predicted_labels = selected_model.named_steps["classifier"].classes_[probability_matrix.argmax(axis=1)]
    manual_cases = [
        {
            "text": text,
            "expected": expected,
            "predicted": str(predicted),
            "confidence": float(probabilities.max()),
            "correct": bool(expected == predicted),
        }
        for text, expected, predicted, probabilities in zip(
            curated_test["text"],
            curated_test["intent"],
            predicted_labels,
            probability_matrix,
            strict=True,
        )
    ]

    metrics = {
        "dataset": "luigicfilho/sid plus project-authored examples",
        "data_note": (
            "SID is synthetic and multilingual; curated delivery examples are "
            "also synthetic and were authored for this project."
        ),
        "rows": {
            "sid_train": len(source_train),
            "curated_train": len(curated_train),
            "sid_validation": len(source_validation),
            "curated_validation": len(curated_validation),
            "sid_test": len(source_test),
            "curated_test": len(curated_test),
        },
        "labels": TARGET_NAMES,
        "curated_training_weight": CURATED_WEIGHT,
        "selected_model": selected_name,
        "curated_validation_macro_f1": validation_scores,
        "sid_test": source_metrics,
        "curated_test": curated_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": selected_model,
            "labels": TARGET_NAMES,
            "selected_model": selected_name,
            "sid_test_metrics": source_metrics,
            "curated_test_metrics": curated_metrics,
        },
        MODEL_PATH,
    )
    (REPORTS_PATH / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_PATH / "curated_test_predictions.json").write_text(
        json.dumps(manual_cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
