from fastapi import FastAPI, Query, File, UploadFile, HTTPException
from pydantic import BaseModel
from app.db import init_db, add_metric, get_metrics, get_training_details
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from app.cifar_10_model_testing import evaluate_cifar10_model
from app.cifar_10_model_train import train_cifar10_model
from app.inference import load_and_predict_image
import numpy as np
from typing import Optional

app = FastAPI()
init_db()

class Features(BaseModel):
    features: list[float]

@app.get("/")
async def welcome():
    return {"message": "Welcome to the Model Training Microservice!"}

@app.post("/train")
async def train_model(epochs: int = Query(10, description="Number of epochs"),
    batch_size: int = Query(64, description="Batch size"),
    validation_split: float = Query(0.1, description="Validation split ratio"),
    learning_rate: float = Query(0.001, description="Learning rate")):
    try:
        training_metrics=train_cifar10_model(epochs,batch_size,validation_split,learning_rate)
        return "Trained Model successfully",training_metrics
    except:
        return "Please Enter epochs,batch_size,validation_split,learning_rate correct manner in the Query Params."

@app.post("/predict")
async def predict():
    prediction = load_and_predict_image()
    return {"prediction": prediction}
    
@app.post("/evaluate")
async def evaluate():
    return evaluate_cifar10_model()


@app.get("/metrics")
async def metrics(n: Optional[int] = None):
    try:
        metrics = get_metrics(n)
        return metrics
    except Exception as e:
        raise HTTPException(detail=f"Error occured: {str(e)}")

@app.get("/training-details")
async def fetch_training_details(n: int = None):
    try:
        details = get_training_details(n)
        if not details:
            raise HTTPException(status_code=404, detail="No training details found.")
        return {"training_details": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

