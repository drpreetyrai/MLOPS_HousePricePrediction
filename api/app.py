import os 
import joblib 
import pandas as pd 
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 

app = FastAPI() 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, "../models/house_price_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl") 


if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found. Please train the model first.") 

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Please train the model first.") 

# start unloading the trained model and scaler 
model = joblib.load(MODEL_PATH) 
scaler = joblib.load(SCALER_PATH) 

# class HouseFeatures(BaseModel):
#     Bedrooms: float 
#     Space: float 
#     Room: float 
#     Lot: float 
#     Tax: float 
#     Bathroom: float 
#     Garage: float 
#     Condition: float 

class HouseFeatures(BaseModel):
    Bedrooms: float 
    Space: float 
    Room: float 
    Lot: float 
    Tax: float 
    Bathroom: float 
    Garage: float 
    Condition: float 


@app.get("/") 
def home():
    return {"message": "Welcome to the House Price Prediction API"}  


@app.post("/predict/") 
def predict_price(features: HouseFeatures): 
    try:
        input_data = pd.DataFrame([features.dict()])  # convert pydantic model to pandas DataFrame
        scaled_data = scaler.transform(input_data)  # scale the input data 
        prediction = model.predict(scaled_data)  # make prediction
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
# uvicorn app:app

# uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

