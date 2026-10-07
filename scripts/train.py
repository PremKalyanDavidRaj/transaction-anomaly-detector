"""Train and evaluate an Isolation Forest anomaly detector."""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FEATURES = [
    "amount",
    "hour",
    "distance_from_home",
    "transactions_last_hour",
    "account_age_days",
    "device_trust_score",
    "international",
]


def evaluate(frame, predictions, scores):
    actual = frame["is_anomaly"].astype(int)

    # Isolation Forest returns -1 for anomalies and 1 for normal rows.
    predicted = (predictions == -1).astype(int)

    # Larger anomaly scores should mean more suspicious.
    anomaly_score = -scores

    report = classification_report(
        actual,
        predicted,
        output_dict=True,
        zero_division=0,
    )

    return {
        "rows": len(frame),
        "actual_anomalies": int(actual.sum()),
        "predicted_anomalies": int(predicted.sum()),
        "precision": float(precision_score(actual, predicted)),
        "recall": float(recall_score(actual, predicted)),
        "f1": float(f1_score(actual, predicted)),
        "average_precision": float(
            average_precision_score(actual, anomaly_score)
        ),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
        "classification_report": report,
    }


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=300,
            contamination=0.03,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    started = time.perf_counter()

    # Do not use is_anomaly while fitting. This is unsupervised training.
    model.fit(train[FEATURES])

    training_seconds = time.perf_counter() - started

    predictions = model.predict(test[FEATURES])
    scores = model.decision_function(test[FEATURES])

    metrics = evaluate(test, predictions, scores)
    metrics.update({
        "model": "IsolationForest",
        "training_rows": len(train),
        "test_rows": len(test),
        "features": FEATURES,
        "contamination": 0.03,
        "n_estimators": 300,
        "random_state": 42,
        "training_seconds": round(training_seconds, 3),
    })

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "anomaly_detector.joblib")

    (REPORT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions_frame = test.copy()
    predictions_frame["predicted_anomaly"] = (
        predictions == -1
    ).astype(int)
    predictions_frame["anomaly_score"] = (-scores).round(6)

    predictions_frame.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    print(json.dumps({
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "average_precision": metrics["average_precision"],
        "confusion_matrix": metrics["confusion_matrix"],
        "training_seconds": metrics["training_seconds"],
    }, indent=2))

    print("\nModel saved to models/anomaly_detector.joblib")
    print("Predictions saved to reports/test_predictions.csv")


if __name__ == "__main__":
    main()
