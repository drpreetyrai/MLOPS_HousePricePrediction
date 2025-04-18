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

model = joblib.load(MODEL_PATH) 


