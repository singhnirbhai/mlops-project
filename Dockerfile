# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Git install
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone your repo
RUN git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git .

# Install deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install fastapi uvicorn scikit-learn pandas numpy joblib

# 🔥 Train during build (creates .pkl)
RUN python3 train.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
