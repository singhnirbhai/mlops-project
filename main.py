from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Earthquake Death Category Predictor")

# Load trained model
model = joblib.load("earthquake_model.pkl")

# Input schema (matches train.py → lat, lon)
class EarthquakeInput(BaseModel):
    lat: float
    lon: float

# Label mapping (must match training labels)
label_map = {
    0: "0 Deaths",
    1: "1-50 Deaths",
    2: "51-100 Deaths",
    3: "101-1000 Deaths",
    4: ">1001 Deaths"
}

@app.get("/")
def home():
    return {"message": "Earthquake Prediction API is live"}

@app.post("/predict")
def predict(data: EarthquakeInput):
    input_data = np.array([[data.lat, data.lon]])

    prediction = model.predict(input_data)[0]
    result = label_map.get(prediction, "Unknown")

    return {
        "prediction": result
    }
