from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np
from typing import Dict
from collections import deque
import os

app = FastAPI(title="AeroDrift Predictive Maintenance API", description="Real-time RUL prediction for turbofan engines.")

# --- Global State & Model ---
model_path = os.path.join(os.path.dirname(__file__), '../../models/lgb_model.txt')
booster = None

class EngineState:
    def __init__(self, window_size=15):
        # deque automatically pops old values when maxlen is reached!
        self.sensor_4 = deque(maxlen=window_size)
        self.sensor_11 = deque(maxlen=window_size)

# In-memory store: engine_id -> EngineState
engine_states: Dict[int, EngineState] = {}

@app.on_event("startup")
def load_model():
    global booster
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}. Please train the model first.")
    booster = lgb.Booster(model_file=model_path)
    print("LightGBM model loaded successfully.")

# --- Request/Response Models ---
class TelemetryPayload(BaseModel):
    engine_id: int
    cycle: int
    sensor_4: float
    sensor_11: float

class PredictionResponse(BaseModel):
    engine_id: int
    cycle: int
    status: str
    message: str
    RUL: float | None = None

# --- Endpoints ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(data: TelemetryPayload):
    # 1. Initialize state for new engines
    if data.engine_id not in engine_states:
        engine_states[data.engine_id] = EngineState(window_size=15)
        
    state = engine_states[data.engine_id]
    
    # 2. Append new readings to the rolling window buffer
    state.sensor_4.append(data.sensor_4)
    state.sensor_11.append(data.sensor_11)
    
    # 3. Check if we have enough data (15 cycles) to make a prediction
    if len(state.sensor_4) < 15:
        return PredictionResponse(
            engine_id=data.engine_id,
            cycle=data.cycle,
            status="buffering",
            message=f"Collecting telemetry... {len(state.sensor_4)}/15 cycles buffered.",
            RUL=None
        )
        
    # 4. Compute rolling features
    # Note: Polars/Pandas rolling_std uses ddof=1 (sample standard deviation) by default
    s4_mean = np.mean(state.sensor_4)
    s11_mean = np.mean(state.sensor_11)
    s4_std = np.std(state.sensor_4, ddof=1) 
    s11_std = np.std(state.sensor_11, ddof=1)
    
    # 5. Prepare feature vector
    # Order MUST match training: cycle, sensor_4, sensor_11, s4_mean, s11_mean, s4_std, s11_std
    features = np.array([[
        data.cycle,
        data.sensor_4,
        data.sensor_11,
        s4_mean,
        s11_mean,
        s4_std,
        s11_std
    ]])
    
    # 6. Predict
    predicted_rul = booster.predict(features)[0]
    
    # Optional: If predicted RUL is extremely high, we can cap it at 130 to match our training logic,
    # but the LightGBM model is already trained on capped data, so it naturally tops out around 130.
    
    return PredictionResponse(
        engine_id=data.engine_id,
        cycle=data.cycle,
        status="success",
        message="RUL predicted successfully.",
        RUL=round(float(predicted_rul), 2)
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": booster is not None}
