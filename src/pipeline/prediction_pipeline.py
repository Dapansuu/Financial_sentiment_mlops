import os
import sys
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import tensorflow as tf

from src.utils.exception import CustomException
from src.utils.logger import logging


@dataclass
class PredictionPipelineConfig:
    model_path: str = os.path.join("artifacts", "inference_model.keras")


class PredictionPipeline:
    def __init__(self):
        try:
            self.config = PredictionPipelineConfig()

            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(
                    f"Inference model not found at: {self.config.model_path}"
                )

            self.model = tf.keras.models.load_model(self.config.model_path)

            self.idx_to_label = {
                0: "neutral",
                1: "positive",
                2: "negative",
            }

            logging.info("PredictionPipeline initialized successfully")

        except Exception as e:
            logging.exception("Failed to initialize PredictionPipeline")
            raise CustomException(e, sys)

    def predict(self, texts: Union[str, List[str]]):
        try:
            if isinstance(texts, str):
                texts = [texts]

            if not isinstance(texts, list) or len(texts) == 0:
                raise ValueError("Input must be a non-empty string or list of strings")

            texts = [str(text).strip() for text in texts]

            if any(text == "" for text in texts):
                raise ValueError("Input contains empty text")

            input_data = np.array(texts, dtype=object).reshape(-1, 1)

            predictions = self.model.predict(input_data, verbose=0)
            predicted_indices = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)

            results = []
            for text, pred_idx, conf, probs in zip(
                texts, predicted_indices, confidences, predictions
            ):
                results.append(
                    {
                        "text": text,
                        "predicted_label": self.idx_to_label[int(pred_idx)],
                        "confidence": float(conf),
                        "probabilities": {
                            "neutral": float(probs[0]),
                            "positive": float(probs[1]),
                            "negative": float(probs[2]),
                        },
                    }
                )

            logging.info("Prediction completed successfully")
            return results

        except Exception as e:
            logging.exception("Prediction failed")
            raise CustomException(e, sys)


def main():
    try:
        predictor = PredictionPipeline()

        sample_texts = [
            "The company posted strong quarterly earnings and growth.",
            "The stock remained unchanged throughout the day.",
            "The firm reported heavy losses and weak guidance.",
        ]

        results = predictor.predict(sample_texts)

        for i, result in enumerate(results, start=1):
            print(f"\nPrediction {i}")
            print(f"Text: {result['text']}")
            print(f"Label: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: {result['probabilities']}")

    except Exception as e:
        logging.exception("Error in prediction pipeline main")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()