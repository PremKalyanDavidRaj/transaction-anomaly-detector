"""Load the detector and score one transaction."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "anomaly_detector.joblib"

FEATURES = [
    "amount",
    "hour",
    "distance_from_home",
    "transactions_last_hour",
    "account_age_days",
    "device_trust_score",
    "international",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python scripts/train.py"
        )
    return joblib.load(MODEL_PATH)


def predict_transaction(model, transaction):
    frame = pd.DataFrame([transaction], columns=FEATURES)

    prediction = int(model.predict(frame)[0])
    raw_score = float(model.decision_function(frame)[0])
    anomaly_score = -raw_score
    is_anomaly = prediction == -1

    if is_anomaly:
        risk_level = "high"
    elif anomaly_score > 0:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "is_anomaly": is_anomaly,
        "risk_level": risk_level,
        "anomaly_score": round(anomaly_score, 6),
        "message": (
            "Review this transaction manually."
            if is_anomaly
            else "No anomaly was detected."
        ),
        "note": (
            "The score is model-specific and is not a probability. "
            "Thresholds require business validation."
        ),
    }
