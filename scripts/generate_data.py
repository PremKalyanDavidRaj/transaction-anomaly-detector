"""Generate reproducible synthetic transaction data for anomaly detection."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

RANDOM_SEED = 42


def make_transactions(rows, anomaly_rate, seed):
    rng = np.random.default_rng(seed)

    amount = rng.lognormal(mean=3.2, sigma=0.8, size=rows)
    hour = rng.integers(0, 24, size=rows)
    distance_from_home = rng.exponential(scale=8, size=rows)
    transactions_last_hour = rng.poisson(lam=1.5, size=rows)
    account_age_days = rng.integers(30, 4000, size=rows)
    device_trust_score = rng.beta(8, 2, size=rows)
    international = rng.binomial(1, 0.08, size=rows)

    labels = np.zeros(rows, dtype=int)
    anomaly_count = int(rows * anomaly_rate)
    anomaly_indices = rng.choice(rows, size=anomaly_count, replace=False)

    labels[anomaly_indices] = 1

    # Inject suspicious behavior into anomalous transactions.
    amount[anomaly_indices] *= rng.uniform(8, 30, size=anomaly_count)
    hour[anomaly_indices] = rng.choice(
        [0, 1, 2, 3, 4], size=anomaly_count
    )
    distance_from_home[anomaly_indices] *= rng.uniform(
        10, 50, size=anomaly_count
    )
    transactions_last_hour[anomaly_indices] += rng.integers(
        8, 20, size=anomaly_count
    )
    device_trust_score[anomaly_indices] *= rng.uniform(
        0.05, 0.35, size=anomaly_count
    )
    international[anomaly_indices] = 1

    frame = pd.DataFrame({
        "amount": amount.round(2),
        "hour": hour,
        "distance_from_home": distance_from_home.round(2),
        "transactions_last_hour": transactions_last_hour,
        "account_age_days": account_age_days,
        "device_trust_score": device_trust_score.round(4),
        "international": international,
        "is_anomaly": labels,
    })

    return frame


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train = make_transactions(
        rows=20000,
        anomaly_rate=0.03,
        seed=RANDOM_SEED,
    )

    test = make_transactions(
        rows=5000,
        anomaly_rate=0.03,
        seed=RANDOM_SEED + 1,
    )

    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"Created {len(train):,} training transactions")
    print(f"Created {len(test):,} test transactions")
    print(f"Training anomalies: {train['is_anomaly'].sum():,}")
    print(f"Test anomalies: {test['is_anomaly'].sum():,}")


if __name__ == "__main__":
    main()
