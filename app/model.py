import mlflow.sklearn
import os

MODEL_URI = "mlruns/latest_model"

if not os.path.exists(MODEL_URI):
    raise RuntimeError("No trained model found. Run train.py first.")

model = mlflow.sklearn.load_model(MODEL_URI)

def predict(text: str) -> int:
    return int(model.predict([text])[0])
