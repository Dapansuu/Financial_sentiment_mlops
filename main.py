from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import tensorflow as tf
import numpy as np
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
FRONTEND_DIST = "frontend_dist"

IDX_TO_LABEL = {
    0: "neutral",
    1: "positive",
    2: "negative"
}

class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

model = None


def load_local_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
    print("Model input shape:", model.input_shape)
    print("Model input dtype:", model.inputs[0].dtype)


@app.on_event("startup")
def startup_event():
    try:
        load_local_model()
    except Exception as e:
        print(f"Model load failed: {e}")


@app.get("/api")
def root():
    return {"message": "Sentiment backend is running"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_path": MODEL_PATH
    }


def prepare_model_inputs(input_texts: List[str]):
    input_shape = model.input_shape

    if isinstance(input_shape, tuple):
        if len(input_shape) == 1:
            return tf.constant(input_texts, dtype=tf.string)
        if len(input_shape) == 2 and input_shape[-1] == 1:
            return tf.constant(input_texts, dtype=tf.string)[:, tf.newaxis]

    return tf.constant(input_texts, dtype=tf.string)


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_texts = []

    if request.text and request.text.strip():
        input_texts = [request.text.strip()]
    elif request.texts:
        input_texts = [
            t.strip() for t in request.texts
            if isinstance(t, str) and t.strip()
        ]

    if not input_texts:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'texts'")

    try:
        model_inputs = prepare_model_inputs(input_texts)
        raw_predictions = model.predict(model_inputs, verbose=0)

        results = []

        for text, pred in zip(input_texts, raw_predictions):
            pred = np.array(pred).astype(float).flatten()

            if pred.size > 1:
                pred_idx = int(np.argmax(pred))
                results.append({
                    "text": text,
                    "label": IDX_TO_LABEL.get(pred_idx, str(pred_idx)),
                    "confidence": float(pred[pred_idx]),
                    "scores": pred.tolist()
                })
            else:
                score = float(pred[0])
                label = "positive" if score >= 0.5 else "negative"
                confidence = score if score >= 0.5 else 1.0 - score

                results.append({
                    "text": text,
                    "label": label,
                    "confidence": confidence,
                    "scores": [1.0 - score, score]
                })

        return {"predictions": results}

    except HTTPException:
        raise
    except Exception as e:
        print("Prediction error:", repr(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Serve React build
if os.path.exists(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def serve_root():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        requested_path = os.path.join(FRONTEND_DIST, full_path)
        index_path = os.path.join(FRONTEND_DIST, "index.html")

        if os.path.exists(requested_path) and os.path.isfile(requested_path):
            return FileResponse(requested_path)

        return FileResponse(index_path)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)