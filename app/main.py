import time
import logging
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.model import predict
from app.drift import DriftDetector

app = FastAPI()

# configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# initialize drift detector
drift_detector = DriftDetector()

class InputText(BaseModel):
    text: str

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        latency = (time.time() - start_time) * 1000
        logging.info(f"{request.method} {request.url.path} - {latency:.2f} ms")
        return response
    except Exception as e:
        logging.error(f"Error in {request.method} {request.url.path}: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

@app.post("/predict")
def get_prediction(input: InputText):
    label = predict(input.text)
    kl_div = drift_detector.check_drift([input.text])  # drift detection
    logging.info(f"Drift KL-divergence: {kl_div:.4f}")
    return {"prediction": label, "drift_score": kl_div}
