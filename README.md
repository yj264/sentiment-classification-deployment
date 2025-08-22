# sentiment-classification-deployment
This project demonstrates an end-to-end ML deployment pipeline for sentiment classification using the IMDB dataset.

### Features
- **Model Training & Experiment Tracking**
  - Logistic Regression + TF-IDF on IMDB reviews
  - Experiment tracking with MLflow
  - Auto-save latest model for serving
- **Inference Service**
  - REST API built with FastAPI
  - Containerized with Docker
- **Monitoring**
  - Latency & error logging
  - Simple drift detection based on KL-divergence between training and serving data distributions

