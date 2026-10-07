import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import api


FEATURES = [
    "amount",
    "hour",
    "distance_from_home",
    "transactions_last_hour",
    "account_age_days",
    "device_trust_score",
    "international",
]


@pytest.fixture
def client(monkeypatch):
    training_rows = [
        [35, 14, 2, 1, 1200, 0.95, 0],
        [42, 12, 4, 1, 900, 0.90, 0],
        [28, 16, 1, 2, 1500, 0.98, 0],
        [8500, 2, 450, 15, 120, 0.08, 1],
        [9200, 3, 600, 18, 80, 0.05, 1],
        [7800, 1, 300, 12, 150, 0.10, 1],
    ]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=50,
            contamination=0.33,
            random_state=42,
        )),
    ])

    model.fit(training_rows)

    monkeypatch.setattr(api, "load_model", lambda: model)

    with TestClient(api.app) as test_client:
        yield test_client


def valid_transaction():
    return {
        "amount": 35,
        "hour": 14,
        "distance_from_home": 2,
        "transactions_last_hour": 1,
        "account_age_days": 1200,
        "device_trust_score": 0.95,
        "international": 0,
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_score_response_shape(client):
    response = client.post("/score", json=valid_transaction())

    assert response.status_code == 200

    result = response.json()
    assert isinstance(result["is_anomaly"], bool)
    assert result["risk_level"] in {"low", "medium", "high"}
    assert isinstance(result["anomaly_score"], float)
    assert "message" in result


def test_suspicious_transaction_is_flagged(client):
    class DeterministicAnomalyModel:
        def predict(self, frame):
            return [-1]

        def decision_function(self, frame):
            return [-0.75]

    from src import api
    api.app.state.model = DeterministicAnomalyModel()

    suspicious = {
        "amount": 8500,
        "hour": 2,
        "distance_from_home": 450,
        "transactions_last_hour": 15,
        "account_age_days": 120,
        "device_trust_score": 0.08,
        "international": 1,
    }

    response = client.post("/score", json=suspicious)

    assert response.status_code == 200
    assert response.json()["is_anomaly"] is True
    assert response.json()["risk_level"] == "high"


@pytest.mark.parametrize(
    "field,value",
    [
        ("amount", 0),
        ("hour", 24),
        ("distance_from_home", -1),
        ("transactions_last_hour", -1),
        ("device_trust_score", 2),
        ("international", 2),
    ],
)
def test_invalid_transaction_values(client, field, value):
    transaction = valid_transaction()
    transaction[field] = value

    response = client.post("/score", json=transaction)

    assert response.status_code == 422


def test_missing_field(client):
    transaction = valid_transaction()
    del transaction["amount"]

    response = client.post("/score", json=transaction)

    assert response.status_code == 422
