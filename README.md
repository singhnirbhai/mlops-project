This project helps you learn **Building and Deploying an ML Model** using a simple and real-world use case: predicting whether a  Earthquake death based on. We’ll go from:

- ✅ Model Training
- ✅ Building the Model locally
- ✅ API Deployment with FastAPI
- ✅ Dockerization
- ✅ Kubernetes Deployment

---

## 📊 Problem Statement

Predict if a Earthquake death based on:
- country
- Year
- longitude
- latutude

We use a Random Forest Classifier trained on the **Earthquake death Dataset**.

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/singhnirbhai/mlops-project.git
cd mlops-project
```

### 2. Create Virtual Environment

```
python3 -m venv .mlops
source .mlops/bin/activate
```

### 3. Install Dependencies

```
## putt this inside the requirement.txt file
fastapi
uvicorn[standard]
scikit-learn
pandas
joblib

```
pip install -r requirements.txt
```


## Train the Model

```
python train.py
```

## Run the API Locally

```
uvicorn main:app --reload
```

## Dockerize the API

### Build the Docker Image

```
docker build -t diabetes-prediction-model .
```

### Run the Container

```
docker run -p 8000:8000 diabetes-prediction-model
```

## Deploy to Kubernetes

```
kubectl apply -f diabetes-prediction-model-deployment.yaml
```

🙌 Credits

Created by `NIrbhay singh`
