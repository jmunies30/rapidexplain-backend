# RAPIDEXPLAIN — model API starter

This backend loads the exact LightGBM model + RobustScaler bundle embedded in the supplied Google Colab notebook.

## Files
- `main.py` — FastAPI API with `GET /health` and `POST /predict`
- `model_bundle.joblib` — original serialized model/scaler bundle extracted from the notebook
- `requirements.txt` — Python dependencies

## Run locally
Use Python 3.11 if possible.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Check `http://127.0.0.1:8000/health`.

## POST /predict
Send JSON containing the original raw input columns:
`Time`, `V1` ... `V28`, `Amount`.

Example shape (values below are placeholders, not a real transaction):
```json
{
  "transaction": {
    "Time": 0,
    "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0,
    "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0,
    "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0,
    "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0,
    "Amount": 10
  }
}
```

Response:
```json
{
  "decision": "NOT FRAUD",
  "fraud_probability": 0.01,
  "threshold": 0.3476448648866197,
  "model": "LightGBM"
}
```
The example response is illustrative only. Actual output depends on the submitted feature values.

## Important input limitation
The model was trained on anonymized PCA-style features `V1`–`V28`, plus `Time` and `Amount`. A normal card transaction description, card number, merchant name, or location cannot be converted into these features by this model. Inputs must already be produced by the same feature-generation pipeline as the original dataset. Never send real card numbers, CVVs, PINs, or payment credentials.

## Deployment
Deploy this folder as a Python web service (for example, on Render). Set the start command to:
`uvicorn main:app --host 0.0.0.0 --port $PORT`

After deployment, set the Bolt frontend's API base URL to the deployed service URL. Restrict CORS in `main.py` to your exact frontend domain before public production use.
