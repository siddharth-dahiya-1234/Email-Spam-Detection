import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

model_path = os.getenv("MODEL_PATH", "./spam_classifier.pkl")
vectorizer_path = os.getenv("VECTORIZER_PATH", "./tfidf_vectorizer.pkl")

# lazy-loaded globals
model = None
vectorizer = None

def load_model_and_vectorizer():
    """Lazy-load the model and vectorizer. Raises FileNotFoundError or ValueError on problems."""
    global model, vectorizer
    if model is not None and vectorizer is not None:
        return

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
        if not hasattr(loaded_model, "predict"):
            raise ValueError("Loaded model does not have a predict method.")
        if not hasattr(loaded_model, "predict_proba"):
            raise ValueError("Loaded model does not have a predict_proba method.")
        model = loaded_model

    with open(vectorizer_path, "rb") as f:
        loaded_vectorizer = pickle.load(f)
        if not hasattr(loaded_vectorizer, "transform"):
            raise ValueError("Loaded vectorizer does not have a transform method.")
        vectorizer = loaded_vectorizer


app = FastAPI()
origins = [ "http://localhost:5500", "http://127.0.0.1:5500" ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

class Prediction(BaseModel):
    label: str
    probability: float

class PredictionResponse(BaseModel):
    message: str
    predictions: List[Prediction]

@app.get("/")
def root():
    return {"message": "Welcome to the Spam Classifier API!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(message: Message):
    # ensure model & vectorizer are available
    try:
        load_model_and_vectorizer()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    text = message.text
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Text must be a string.")

    text_vectorized = vectorizer.transform([text])

    prediction = model.predict(text_vectorized)
    prediction_proba = model.predict_proba(text_vectorized)

    # normalize types
    label = "spam" if int(prediction[0]) == 1 else "ham"
    probability = float(np.max(prediction_proba[0]))

    return PredictionResponse(
        message="Prediction successful",
        predictions=[Prediction(label=label, probability=probability)]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)