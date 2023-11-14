import uvicorn
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
from typing import List
from pydantic import BaseModel, Field

# create the app object
app = FastAPI()

model = load_model('wkts_NN_model.h5')  # Load your saved model
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

runs_model = load_model('runs_NN_model.h5')
with open('scaler_runs.pkl', 'rb') as scaler_runs_file:
    runs_scaler = pickle.load(scaler_runs_file)

class WicketsInput(BaseModel):
    Inns: int
    Overs: float
    Mdns: int
    Econ: float
    Four_wickets: int 
    Five_wickets: int 

class RunsInput(BaseModel):
    Inns: int
    Ave: float
    SR: float
    Hundreds: int
    Fifties: int
    Fours: int
    Sixes: int

@app.get('/')
def index():
    return {"message": "Hello! Welcome to Wickets and Runs Prediction API."}


@app.post("/predict_wickets", response_model=dict)
def predict_wkts(player_input: WicketsInput):
    input_data = pd.DataFrame([player_input.dict()])
    input_scaled = scaler.transform(input_data)
    predicted_wkts = model.predict(input_scaled)
    predicted_wkts = np.ceil(predicted_wkts).astype(int)
    predicted_wkts_list = predicted_wkts.tolist()
    return {"predicted_wkts": predicted_wkts_list}



@app.post("/predict_runs", response_model=dict)
def predict_runs(player_input: RunsInput):
    input_data = pd.DataFrame([player_input.dict()])
    input_scaled = runs_scaler.transform(input_data)
    predicted_runs = runs_model.predict(input_scaled)
    predicted_runs = np.ceil(predicted_runs).astype(int)
    predicted_runs_list = predicted_runs.tolist()
    return {"predicted_runs": predicted_runs_list}


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)