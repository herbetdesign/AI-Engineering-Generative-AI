"""Publish a trained pipeline and its honest evaluation to Hugging Face."""

import argparse
import json

from huggingface_hub import HfApi

from foodintent_ml.train import MODEL_PATH, REPORTS_PATH


def main() -> None:
    """Upload model, test metrics, and a model card using the cached login."""
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="e.g. username/foodintent-ml")
    args = parser.parse_args()

    metrics_path = REPORTS_PATH / "metrics.json"
    if not MODEL_PATH.is_file() or not metrics_path.is_file():
        raise SystemExit("Treine primeiro com: python -m foodintent_ml.train")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo="foodintent_pipeline.joblib",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload trained FoodIntent pipeline",
    )
    card = (
        "---\nlicense: mit\ntags:\n- text-classification\n- scikit-learn\n"
        "- portuguese\nlibrary_name: sklearn\n---\n\n"
        "# FoodIntent ML\n\n"
        "Experimental TF-IDF intent classifier. Every prediction requires "
        "human review.\n\n"
        "## Evaluation\n\n"
        f"- Selected model: {metrics['selected_model']}\n"
        f"- Curated test macro F1: "
        f"{metrics['curated_test']['macro_f1']:.4f}\n"
        f"- SID test macro F1: {metrics['sid_test']['macro_f1']:.4f}\n\n"
        "## Data and limitations\n\n"
        "Both the SID source and the project-authored delivery supplement are "
        "synthetic. These metrics do not estimate performance on real customers. "
        "Confidence values are not calibrated. Do not automate customer-impacting "
        "decisions.\n\n"
        "Dataset: https://huggingface.co/datasets/luigicfilho/sid\n"
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add model card with measured metrics and limitations",
    )
    api.upload_file(
        path_or_fileobj=metrics_path.read_bytes(),
        path_in_repo="metrics.json",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Publish held-out evaluation metrics",
    )
    print(f"Published: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
