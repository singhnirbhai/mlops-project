from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import numpy as np

app = FastAPI(title="🌍 Earthquake Predictor")

# Load model
model = joblib.load("earthquake_model.pkl")
country_encoder = joblib.load("country_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -------------------------------
# UI PAGE
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Earthquake Predictor</title>
        <style>
            body {
                margin: 0;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                font-family: Arial;
                background: linear-gradient(135deg, #a0d2eb, #a28089);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.2);
                padding: 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                text-align: center;
                width: 350px;
            }
            input {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border-radius: 8px;
                border: none;
            }
            button {
                width: 100%;
                padding: 12px;
                border-radius: 8px;
                border: none;
                background: white;
                color: #333;
                font-weight: bold;
                cursor: pointer;
            }
            button:hover {
                background: #ddd;
            }
            .result {
                margin-top: 20px;
                font-size: 20px;
                font-weight: bold;
                color: #fff;
            }
        </style>
    </head>
    <body>

        <div class="container">
            <h2>🌍 Earthquake Predictor</h2>

            <input id="country" placeholder="Country">
            <input id="year" type="number" placeholder="Year">
            <input id="lat" type="number" step="any" placeholder="Latitude">
            <input id="lon" type="number" step="any" placeholder="Longitude">

            <button onclick="predict()">Predict</button>

            <div class="result" id="result"></div>
        </div>

        <script>
            async function predict() {
                const country = document.getElementById("country").value;
                const year = document.getElementById("year").value;
                const lat = document.getElementById("lat").value;
                const lon = document.getElementById("lon").value;

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        country: country,
                        year: parseInt(year),
                        lat: parseFloat(lat),
                        lon: parseFloat(lon)
                    })
                });

                const data = await response.json();

                document.getElementById("result").innerText =
                    "Prediction: " + (data.prediction || data.error);
            }
        </script>

    </body>
    </html>
    """

# -------------------------------
# API
# -------------------------------
@app.post("/predict")
async def predict_api(request: Request):
    data = await request.json()

    try:
        c = country_encoder.transform([data["country"]])[0]
    except:
        return JSONResponse({"error": "Country not found"})

    input_data = np.array([[c, data["year"], data["lat"], data["lon"]]])
    pred = model.predict(input_data)[0]

    result = label_encoder.inverse_transform([pred])[0]

    return {"prediction": result}
