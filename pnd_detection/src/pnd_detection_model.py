import pickle 
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pump and Dump Detection API",description="API to predict pump and dump schemes in cryptocurrency markets",version="0.1")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
class PredictionRequest(BaseModel): 
    std_rush_order : float
    avg_rush_order : float 
    std_trades : float
    std_volume : float 
    avg_volume : float
    std_price : float
    avg_price : float
    avg_price_max : float
    hour_sin : float 
    hour_cos : float
    minute_sin : float
    minute_cos : float


def load_model(model_path : str = "model.pkl"):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception("Error loading model: " + str(e))

model = load_model()

def get_features_array(request : PredictionRequest) -> np.ndarray: 
    return np.array([request.std_rush_order, request.avg_rush_order,
                      request.std_trades, request.std_volume, request.avg_volume, request.std_price,
                        request.avg_price, request.avg_price_max, request.hour_sin, request.hour_cos,
                          request.minute_sin, request.minute_cos]).reshape(1,-1)


@app.post("/predict")
async def predict(request : PredictionRequest):
    try:
        features = get_features_array(request)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        e.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("pnd_detection_model:app", host="0.0.0.0", port=5040,reload=True)




