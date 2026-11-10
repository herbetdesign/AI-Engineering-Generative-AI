"""Train candidate text classifiers and report held-out test metrics."""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from foodintent_ml.dataset import LABEL_MAP, load_split

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "foodintent_pipeline.joblib"
REPORTS_PATH = ROOT / "reports"
TARGET_NAMES = sorted(set(LABEL_MAP.values()))


def build_candidates() -> dict[str, Pipeline]:
    """Create TF-IDF baselines with two classical classifiers."""
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


def main() -> None:
    """Select by validation macro F1 and evaluate the chosen model on test."""
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")
    if train.empty or validation.empty or test.empty:
        raise ValueError("A filtered dataset split is empty.")

    candidates = build_candidates()
    validation_scores: dict[str, float] = {}
    for name, model in candidates.items():
        model.fit(train["text"], train["intent"])
        prediction = model.predict(validation["text"])
        validation_scores[name] = f1_score(
            validation["intent"], prediction, labels=TARGET_NAMES,
            average="macro", zero_division=0,
        )

    selected_name = max(validation_scores, key=validation_scores.get)
    selected_model = candidates[selected_name]
    predictions = selected_model.predict(test["text"])
    metrics = {
        "dataset": "luigicfilho/sid",
        "data_note": "Synthetic multilingual source; Portuguese subset only.",
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "labels": TARGET_NAMES,
        "selected_model": selected_name,
        "validation_macro_f1": validation_scores,
        "test": {
            "accuracy": accuracy_score(test["intent"], predictions),
            "macro_f1": f1_score(
                test["intent"], predictions, labels=TARGET_NAMES,
                average="macro", zero_division=0,
            ),
            "classification_report": classification_report(
                test["intent"], predictions, labels=TARGET_NAMES,
                output_dict=True, zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                test["intent"], predictions, labels=TARGET_NAMES
            ).tolist(),
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": selected_model,
            "labels": TARGET_NAMES,
            "selected_model": selected_name,
            "test_metrics": metrics["test"],
        },
        MODEL_PATH,
    )
    (REPORTS_PATH / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
