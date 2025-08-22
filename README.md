Sentiment Classification Deployment (FastAPI + MLflow + Docker)

This project demonstrates an end-to-end ML deployment workflow using a sentiment classification model trained on the IMDB dataset.
It covers the full lifecycle from model training and experiment tracking (MLflow), to serving via FastAPI API,
and containerization with Docker, with built-in monitoring (latency, error logging, drift detection).

------------------------------------------------------------
✨ Highlights
------------------------------------------------------------
- End-to-end ML pipeline with training → logging → serving
- FastAPI service exposing real-time predictions
- MLflow experiment tracking & model registry
- Docker containerization for reproducible deployment
- Monitoring pipeline: latency logging, error logging, and drift detection (KL-divergence)

------------------------------------------------------------
📂 Project Structure
------------------------------------------------------------
sentiment-deploy/
│── app/
│   ├── main.py        # FastAPI API (monitoring + drift detection)
│   ├── model.py       # Load latest MLflow model
│   └── drift.py       # Drift detection logic
│
│── train.py           # Train model + log to MLflow
│── requirements.txt   # Dependencies
│── Dockerfile         # Docker build file
│── README.txt

------------------------------------------------------------
⚙️ Setup & Usage
------------------------------------------------------------

1. Install dependencies
   pip install -r requirements.txt

2. Train model & save with MLflow
   python train.py

   - Trains a sentiment classifier on IMDB dataset
   - Logs accuracy and parameters to MLflow
   - Saves the latest model to mlruns/latest_model/
   - Exports training word distribution for drift detection

   Launch MLflow UI:
   mlflow ui
   Visit: http://127.0.0.1:5000

3. Run FastAPI server
   uvicorn app.main:app --reload --port 8000

   Interactive docs: http://127.0.0.1:8000/docs

   Example request:
   curl -X POST "http://127.0.0.1:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "This movie was fantastic!"}'

   Example response:
   {
     "prediction": 1,
     "drift_score": 0.0042
   }

   Logs are stored in logs/app.log

4. Build & Run with Docker
   docker build -t sentiment-api .
   docker run -p 8000:8000 sentiment-api

   API available at http://127.0.0.1:8000/docs

------------------------------------------------------------
📝 Monitoring
------------------------------------------------------------
- Latency & Errors: Logged for each request in logs/app.log
- Drift Detection: KL-divergence between training word distribution and incoming text

Example log:
2025-08-22 12:30:45,123 - INFO - POST /predict - 2.35 ms
2025-08-22 12:30:45,124 - INFO - Drift KL-divergence: 0.0042

------------------------------------------------------------
📦 Tech Stack
------------------------------------------------------------
- ML: scikit-learn, MLflow
- Serving: FastAPI, Uvicorn
- Monitoring: Logging + Drift Detection
- Deployment: Docker

------------------------------------------------------------
✅ Next Steps
------------------------------------------------------------
- Add CI/CD pipeline with GitHub Actions
- Deploy to AWS Fargate or ECS for serverless serving
- Extend monitoring with Prometheus + Grafana

------------------------------------------------------------
📌 Resume-Ready Summary
------------------------------------------------------------
Built an end-to-end ML deployment project: trained sentiment classifier (IMDB), tracked experiments with MLflow,
served model via FastAPI API, containerized with Docker, and implemented monitoring pipeline
(latency, error logging, drift detection).
