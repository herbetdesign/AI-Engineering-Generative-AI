"""Load the public source and locally curated delivery examples."""

from pathlib import Path

import pandas as pd

BASE_URL = "https://huggingface.co/datasets/luigicfilho/sid/resolve/main"
SPLIT_FILES = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
CURATED_SPLITS = {"train", "validation", "test"}
TARGET_NAMES = [
    "order_status",
    "cancellation",
    "payment_issue",
    "refund_request",
    "delivery_problem",
]
LABEL_MAP = {
    "track_order": "order_status",
    "check_order_status": "order_status",
    "cancel_order": "cancellation",
    "report_payment_failure": "payment_issue",
    "dispute_charge": "payment_issue",
    "request_refund": "refund_request",
}


def load_split(split: str) -> pd.DataFrame:
    """Load the SID split and keep unambiguous Portuguese intents."""
    try:
        filename = SPLIT_FILES[split]
    except KeyError as error:
        raise ValueError(f"Unsupported split: {split}") from error

    frame = pd.read_csv(f"{BASE_URL}/{filename}")
    required = {"text", "label", "language"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    frame = frame.loc[frame["language"].eq("pt")].copy()
    frame["intent"] = frame["label"].map(LABEL_MAP)
    frame = frame.dropna(subset=["intent"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame.loc[frame["text"].ne("")]
    frame["origin"] = "sid_synthetic"
    return frame[["text", "intent", "origin"]].reset_index(drop=True)


def load_curated(split: str) -> pd.DataFrame:
    """Load project-authored synthetic examples for delivery-specific language."""
    if split not in CURATED_SPLITS:
        raise ValueError(f"Unsupported curated split: {split}")

    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "data" / "curated" / f"{split}.csv"
    frame = pd.read_csv(path)
    required = {"text", "intent"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Curated data is missing columns: {sorted(missing)}")
    if not set(frame["intent"]).issubset(TARGET_NAMES):
        raise ValueError("Curated data contains an unknown intent.")

    frame["text"] = frame["text"].astype(str).str.strip()
    frame["origin"] = "project_authored_synthetic"
    return frame[["text", "intent", "origin"]].reset_index(drop=True)
