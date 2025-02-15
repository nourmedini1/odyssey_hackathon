
from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError  # For custom object mapping

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import uvicorn
import tensorflow as tf
tf.config.run_functions_eagerly(True) 
from tensorflow.keras.optimizers import Adam

import requests
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Crypto LSTM Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

model = load_model("crypto_lstm_model.h5", custom_objects={'mse': MeanSquaredError()})

scaler = joblib.load("scaler.pkl")


#-------Prediction-----------------------------



SEQ_LENGTH = 30   
PRED_LENGTH = 7   



def load_data():

    df = pd.read_csv("preprocessed_ethereum_data.csv", index_col=0, parse_dates=True)
    df.sort_index(inplace=True)  
    return df

# --- Predict Endpoint ---
@app.get("/predict")
def predict():
    """
    Uses the most recent SEQ_LENGTH days from the preprocessed data to predict the next PRED_LENGTH days
    of the 'Close' price.
    """
    # Load the preprocessed data
    df = load_data()
    data = df.values 
    n_features = data.shape[1]  

    
    data_scaled = scaler.transform(data)

    
    if data_scaled.shape[0] < SEQ_LENGTH:
        return {"error": "Not enough data for prediction."}

    
    last_sequence = data_scaled[-SEQ_LENGTH:]  
    
    last_sequence = last_sequence.reshape(1, SEQ_LENGTH, n_features)

  
    predictions_scaled = model.predict(last_sequence)  
    predictions_scaled = predictions_scaled.reshape(PRED_LENGTH, 1)  


    close_col_index = 3
    dummy_array = np.zeros((PRED_LENGTH, n_features))
  
    dummy_array[:, close_col_index] = predictions_scaled.flatten()

    predictions_full = scaler.inverse_transform(dummy_array)
 
    predictions = predictions_full[:, close_col_index]


    return {"predicted_close_prices": predictions.tolist()}




#------------fine-tunining---------------------------------------------
def fetch_yesterday_data(symbol="ETHUSDT"):
    
    end_time = int(datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000) - 1
    start_time = int((datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)).timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception("Error fetching data from Binance: " + response.text)
    data = response.json()
    
    if len(data) == 0:
        raise Exception("No data returned from Binance for the given period.")
    
   
    kline = data[0]
    
    daily_data = {
        "date": datetime.utcfromtimestamp(kline[0]/1000).strftime("%b %d, %Y"),
        "open": float(kline[1]),
        "high": float(kline[2]),
        "low": float(kline[3]),
        "close": float(kline[4]),
        "volume": float(kline[5])
    }
    return daily_data


def update_historical_data(new_data: dict, csv_path="preprocessed_ethereum_data.csv"):
    import pandas as pd
    
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    new_row = pd.DataFrame({
        "Open": [new_data["open"]],
        "High": [new_data["high"]],
        "Low": [new_data["low"]],
        "Close": [new_data["close"]],
        "Volume": [new_data["volume"]]
    }, index=[pd.to_datetime(new_data["date"], format="%b %d, %Y")])
    
    
    df = pd.concat([df, new_row])
    df.sort_index(inplace=True)
    
    df.to_csv(csv_path)
    return df



def fine_tune_model_with_data(df, scaler, model, seq_length=30, pred_length=7):
    
    data = df.values
    
    scaler.fit(data)
    updated_scaled = scaler.transform(data)
    
    
    def create_sequences_multi_step(data, seq_length, pred_length):
        X, y = [], []
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length:i+seq_length+pred_length, 3])  
        return np.array(X), np.array(y)
    
    X, y = create_sequences_multi_step(updated_scaled, seq_length, pred_length)
    
    
    if X.shape[0] > 100:
        X_finetune = X[-100:]
        y_finetune = y[-100:]
    else:
        X_finetune, y_finetune = X, y
    
    
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=1e-7), loss='mse')
    
    
    model.fit(X_finetune, y_finetune, epochs=2, batch_size=16, verbose=1)
    
    
    model.save("crypto_lstm_model.h5")
    import joblib
    joblib.dump(scaler, "scaler.pkl")
    
    return model, scaler




def daily_finetuning_job():
    try:
        print("Running daily fine-tuning job...")
        new_data = fetch_yesterday_data()
        print("Fetched data:", new_data)
        df = update_historical_data(new_data)
        print("Historical data updated.")
        global model, scaler  
        model, scaler = fine_tune_model_with_data(df, scaler, model)
        print("Model fine-tuned successfully.")
    except Exception as e:
        print("Error during daily fine-tuning job:", e)


scheduler = BackgroundScheduler()
scheduler.add_job(daily_finetuning_job, 'cron', hour=6, minute=50)



if __name__ == "__main__":
    scheduler.start()
    uvicorn.run(app, host="0.0.0.0", port=5020)
