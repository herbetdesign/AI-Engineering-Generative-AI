"""Download and prepare the public SID intent dataset."""

import pandas as pd

BASE_URL = "https://huggingface.co/datasets/luigicfilho/sid/resolve/main"
SPLIT_FILES = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
LABEL_MAP = {
    "track_order": "order_status",
    "check_order_status": "order_status",
    "cancel_order": "cancellation",
    "report_payment_failure": "payment_issue",
    "dispute_charge": "payment_issue",
    "request_refund": "refund_request",
    "report_delivery_issue": "delivery_problem",
}


def load_split(split: str) -> pd.DataFrame:
    """Load one split and filter Portuguese delivery-related messages."""
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
    return frame[["text", "intent"]].reset_index(drop=True)
