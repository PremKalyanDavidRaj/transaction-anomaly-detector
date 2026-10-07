"""FastAPI service for transaction anomaly scoring."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.predict import load_model, predict_transaction


@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()
    yield
    del app.state.model


app = FastAPI(
    title="Transaction Anomaly Detector",
    description="Detects suspicious synthetic banking transactions.",
    version="0.1.0",
    lifespan=lifespan,
)


class TransactionRequest(BaseModel):
    amount: float = Field(gt=0, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    distance_from_home: float = Field(ge=0, le=1_000_000)
    transactions_last_hour: int = Field(ge=0, le=10_000)
    account_age_days: int = Field(ge=0, le=100_000)
    device_trust_score: float = Field(ge=0, le=1)
    international: int = Field(ge=0, le=1)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.post("/score")
def score(body: TransactionRequest, request: Request):
    try:
        return predict_transaction(
            request.app.state.model,
            body.model_dump(),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
