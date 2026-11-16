import math
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load the exact model + scaler + feature list + threshold embedded in the Colab notebook.
bundle = joblib.load("model_bundle.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
FEATURES = list(bundle["features"])
THRESHOLD = float(bundle["threshold_f1"])

RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

app = FastAPI(title="RAPIDEXPLAIN Fraud Detection API", version="1.0.0")

# For a hackathon prototype. Restrict this to your deployed Bolt frontend domain
# before using the API publicly with real users.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "model": bundle.get("model_name", "LightGBM"),
        "required_fields": RAW_COLUMNS,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: dict[str, Any]):
    transaction = payload.get("transaction", payload)

    if not isinstance(transaction, dict):
        raise HTTPException(status_code=400, detail="Send a transaction object.")

    missing = [name for name in RAW_COLUMNS if name not in transaction]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Missing required transaction fields.", "missing": missing},
        )

    values = {}
    for name in RAW_COLUMNS:
        try:
            value = float(transaction[name])
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"{name} must be numeric.")
        if not math.isfinite(value):
            raise HTTPException(status_code=422, detail=f"{name} must be a finite number.")
        values[name] = value

    if values["Amount"] < 0:
        raise HTTPException(status_code=422, detail="Amount cannot be negative.")

    # Match the feature engineering in the user's Colab notebook exactly.
    row = pd.DataFrame([values], columns=RAW_COLUMNS)
    row["Amount_log"] = np.log1p(row["Amount"].clip(lower=0))
    seconds_in_day = 24 * 60 * 60
    time_in_day = np.mod(row["Time"].to_numpy(), seconds_in_day)
    row["Hour_sin"] = np.sin(2 * np.pi * time_in_day / seconds_in_day)
    row["Hour_cos"] = np.cos(2 * np.pi * time_in_day / seconds_in_day)

    # Match the notebook's feature ordering and scaler.
    X = row[FEATURES].astype(float)
    X_scaled = scaler.transform(X)

    fraud_probability = float(model.predict_proba(X_scaled)[0, 1])
    decision = "FRAUD" if fraud_probability >= THRESHOLD else "NOT FRAUD"

    return {
        "decision": decision,
        "fraud_probability": fraud_probability,
        "threshold": THRESHOLD,
        "model": bundle.get("model_name", "LightGBM"),
    }
