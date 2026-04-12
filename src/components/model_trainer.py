import os
import sys
import json
from dataclasses import dataclass

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.utils.exception import CustomException
from src.utils.logger import logging

import yaml

@dataclass
class ModelTrainerConfig:
    transformed_dir: str = os.path.join("dataset", "transformed")
    train_data_path: str = os.path.join("dataset", "transformed", "train")
    test_data_path: str = os.path.join("dataset", "transformed", "test")
    val_data_path: str = os.path.join("dataset", "transformed", "val")

    artifact_dir: str = "artifacts"
    model_path: str = os.path.join("artifacts", "model.keras")
    metrics_path: str = os.path.join("artifacts", "metrics.json")
    cm_path: str = os.path.join("artifacts", "confusion_matrix.png")
    report_path: str = os.path.join("artifacts", "classification_report.txt")

    params = yaml.safe_load(open("params.yaml"))
    vocab_size: int = params["model_trainer"]["vocab_size"]
    sequence_length: int = params["model_trainer"]["sequence_length"]
    embedding_dim: int = params["model_trainer"]["embedding_dim"]
    lstm_units: int = params["model_trainer"]["lstm_units"]
    dense_units: int = params["model_trainer"]["dense_units"]
    learning_rate: float = params["model_trainer"]["learning_rate"]
    batch_size: int = params["model_trainer"]["batch_size"]
    epochs: int = params["model_trainer"]["epochs"]


class ModelTrainer:
    def __init__(self):
        try:
            self.config = ModelTrainerConfig()
            os.makedirs(self.config.artifact_dir, exist_ok=True)
            logging.info("ModelTrainer initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def load_transformed_data(self):
        try:
            for path in [
                self.config.train_data_path,
                self.config.test_data_path,
                self.config.val_data_path,
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Transformed dataset not found at: {path}")

            train_ds = tf.data.Dataset.load(self.config.train_data_path)
            test_ds = tf.data.Dataset.load(self.config.test_data_path)
            val_ds = tf.data.Dataset.load(self.config.val_data_path)

            train_ds = train_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
            val_ds = val_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

            logging.info("Transformed datasets loaded successfully")
            return train_ds, test_ds, val_ds

        except Exception as e:
            logging.exception("Failed while loading transformed datasets")
            raise CustomException(e, sys)

    def build_model(self):
        try:
            logging.info("Building BiLSTM model for vectorized inputs")

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.config.sequence_length,)),
                tf.keras.layers.Embedding(
                    input_dim=self.config.vocab_size,
                    output_dim=self.config.embedding_dim,
                    mask_zero=True,
                ),
                tf.keras.layers.SpatialDropout1D(0.2),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        self.config.lstm_units,
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    )
                ),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.config.dense_units, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(3, activation="softmax"),
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(self.config.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=3,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.config.model_path,
                    save_best_only=True,
                    monitor="val_accuracy",
                ),
            ]

            logging.info("Model compiled successfully")
            return model

        except Exception as e:
            logging.exception("Error while building model")
            raise CustomException(e, sys)

    def _save_confusion_matrix(self, y_true, y_pred):
        try:
            cm = confusion_matrix(y_true, y_pred)
            labels = ["neutral", "positive", "negative"]

            plt.figure(figsize=(8, 6))

            im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix", fontsize=14, pad=20)
            plt.colorbar(im, fraction=0.046, pad=0.04) 

            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45) 
            plt.yticks(tick_marks, labels)
            plt.xlabel("Predicted Label", fontweight='bold')
            plt.ylabel("True Label", fontweight='bold')

            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.savefig(self.config.cm_path, dpi=300) 
            plt.close()

            logging.info("Confusion matrix saved at %s", self.config.cm_path)

        except Exception as e:
            logging.exception("Error while saving confusion matrix")
            raise CustomException(e, sys)

    def _collect_labels_and_predictions(self, model, dataset):
        try:
            y_true = []
            y_pred = []

            for X_batch, y_batch in dataset:
                preds = model.predict(X_batch, verbose=0)
                pred_labels = np.argmax(preds, axis=1)

                y_true.extend(y_batch.numpy())
                y_pred.extend(pred_labels)

            return np.array(y_true), np.array(y_pred)

        except Exception as e:
            logging.exception("Error while collecting predictions")
            raise CustomException(e, sys)

    def model_trainer(self):
        try:
            logging.info("Starting model trainer")

            train_ds, test_ds, val_ds = self.load_transformed_data()
            model = self.build_model()

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True,
            )

            mlflow.set_experiment("financial_sentiment_bilstm")

            with mlflow.start_run(run_name="bilstm_training"):
                logging.info("MLflow run started")

                mlflow.log_params({
                    "model_type": "BiLSTM",
                    "optimizer": "Adam",
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "vocab_size": self.config.vocab_size,
                    "sequence_length": self.config.sequence_length,
                    "embedding_dim": self.config.embedding_dim,
                    "lstm_units": self.config.lstm_units,
                    "dense_units": self.config.dense_units,
                })

                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=self.config.epochs,
                    callbacks=[early_stopping],
                    verbose=1,
                )

                logging.info("Model training completed")

                y_true, y_pred = self._collect_labels_and_predictions(model, test_ds)

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
                    "best_val_loss": float(min(history.history.get("val_loss", [0.0]))),
                }

                with open(self.config.metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)

                mlflow.log_metrics(metrics)

                for i, train_loss in enumerate(history.history.get("loss", [])):
                    mlflow.log_metric("train_loss_epoch", float(train_loss), step=i)

                for i, val_loss in enumerate(history.history.get("val_loss", [])):
                    mlflow.log_metric("val_loss_epoch", float(val_loss), step=i)

                for i, train_acc in enumerate(history.history.get("accuracy", [])):
                    mlflow.log_metric("train_accuracy_epoch", float(train_acc), step=i)

                for i, val_acc in enumerate(history.history.get("val_accuracy", [])):
                    mlflow.log_metric("val_accuracy_epoch", float(val_acc), step=i)

                model.save(self.config.model_path)
                logging.info("Model saved to %s", self.config.model_path)

                self._save_confusion_matrix(y_true, y_pred)
                mlflow.log_artifact(self.config.metrics_path, artifact_path="metrics")
                mlflow.log_artifact(self.config.cm_path, artifact_path="plots")

                report = classification_report(y_true, y_pred)
                with open(self.config.report_path, "w") as f:
                    f.write(report)

                mlflow.log_artifact(self.config.report_path, artifact_path="reports")

                mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path="model",
                )

                logging.info("MLflow logging completed successfully")

            return metrics

        except Exception as e:
            logging.exception("Error during model training")
            raise CustomException(e, sys)


def main():
    trainer = ModelTrainer()
    metrics = trainer.model_trainer()
    print(metrics)


if __name__ == "__main__":
    main()