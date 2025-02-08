from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing"
        )
    
    expected_key = os.getenv("API_KEY", "dev-api-key-12345")
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

model = None

def load_model():
    global model
    if model is None:
        model = joblib.load('models/model.pkl')
    return model

class InputData(BaseModel):
    Gender: str
    Age: int
    HasDrivingLicense: int
    RegionID: float
    Switch: int
    PastAccident: str
    AnnualPremium: float

@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}

@app.get("/health")
async def health_check():
    try:
        load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_version": 1
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_loaded": False
        }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
async def predict(input_data: InputData, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    
    try:
        model = load_model()
        df = pd.DataFrame([input_data.model_dump().values()], 
                          columns=input_data.model_dump().keys())
        pred = model.predict(df)
        
        prediction_counter.inc()
        prediction_latency.observe(time.time() - start_time)
        
        return {"predicted_class": int(pred[0])}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )



