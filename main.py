from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np
import tensorflow as tf
import os

app = FastAPI(title="Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join("artifacts", "inference_model.keras")

IDX_TO_LABEL = {
    0: "neutral",
    1: "positive",
    2: "negative"
}

class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    print("Model input shape:", model.input_shape)
    print("Model input dtype:", model.inputs[0].dtype)
except Exception as e:
    model = None
    print(f"Error loading model from {MODEL_PATH}: {e}")

@app.get("/")
def root():
    return {"message": "Sentiment backend is running with local model"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_path": MODEL_PATH,
    }

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_texts = []

    if request.text and request.text.strip():
        input_texts = [request.text.strip()]
    elif request.texts:
        input_texts = [t.strip() for t in request.texts if isinstance(t, str) and t.strip()]

    if not input_texts:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'texts'")

    try:
        # Pass raw text directly.
        # Since TextVectorization is inside the model, do NOT manually reshape to sequence length.
        model_inputs = tf.constant(input_texts)

        predictions = model.predict(model_inputs, verbose=0)

        results = []

        for text, pred in zip(input_texts, predictions):
            pred = pred.tolist() if hasattr(pred, "tolist") else pred

            if isinstance(pred, list) and len(pred) > 1:
                pred_idx = int(np.argmax(pred))
                results.append({
                    "text": text,
                    "label": IDX_TO_LABEL.get(pred_idx, str(pred_idx)),
                    "confidence": float(pred[pred_idx]),
                    "scores": [float(x) for x in pred]
                })
            else:
                score = float(pred[0] if isinstance(pred, (list, np.ndarray)) else pred)
                label = "positive" if score >= 0.5 else "negative"
                confidence = score if score >= 0.5 else 1.0 - score

                results.append({
                    "text": text,
                    "label": label,
                    "confidence": confidence,
                    "score": score
                })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)