import os
import sys
import json
import tempfile
import seaborn as sns
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
from sklearn.model_selection import train_test_split

from src.utils.exception import CustomException
from src.utils.logger import logging


class ModelTrainer:
    def __init__(self):
        self.artifact_dir = "artifacts"
        self.model_path = os.path.join(self.artifacts, "model.keras")
        self.metrics_path = os.path.join(self.artifacts, "metrics.json")
        self.cm_path = os.path.join(self.artifacts, "confusion_matrix.png")

        os.makedirs(self.artifact_dir, exist_ok=True)

    def build_model(self, vectorizer, vocab_size: int = 12000, sequence_length: int = 40):
        try:
            logging.info("Building BiLSTM model")

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(1,), dtype=tf.string),
                vectorizer,
                tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=16,
                    mask_zero=True,
                ),
                tf.keras.layers.SpatialDropout1D(0.2),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        24,
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    )
                ),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(3, activation="softmax"),
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            logging.info("Model compiled successfully")
            return model

        except Exception as e:
            logging.exception("Error while building model")
            raise CustomException(e, sys)

    def _save_confusion_matrix(self, y_true, y_pred):
        try:
            cm = confusion_matrix(y_true, y_pred)
            fig = plt.figure(figsize=(8, 6))
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False)
            plt.show()
            plt.savefig(self.cm_path)

            logging.info("Confusion matrix saved at %s", self.cm_path)

        except Exception as e:
            logging.exception("Error while saving confusion matrix")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X, y, vectorizer):
        try:
            logging.info("Starting model trainer")

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.1,
                random_state=42,
                stratify=y_train,
            )

            batch_size = 32
            epochs = 5
            vocab_size = 12000
            sequence_length = 40

            model = self.build_model(
                vectorizer=vectorizer,
                vocab_size=vocab_size,
                sequence_length=sequence_length,
            )

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
                    "learning_rate": 0.001,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "vocab_size": vocab_size,
                    "sequence_length": sequence_length,
                    "embedding_dim": 16,
                    "lstm_units": 24,
                    "dense_units": 16,
                    "test_size": 0.2,
                    "val_size_from_train": 0.1,
                })

                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=1,
                )

                logging.info("Model training completed")

                y_probs = model.predict(X_test)
                y_pred = np.argmax(y_probs, axis=1)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
                    "best_val_loss": float(min(history.history.get("val_loss", [0.0]))),
                }

                with open(self.metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)

                logging.info("Metrics saved to %s", self.metrics_path)

                mlflow.log_metrics(metrics)

                for i, train_loss in enumerate(history.history.get("loss", [])):
                    mlflow.log_metric("train_loss_epoch", float(train_loss), step=i)

                for i, val_loss in enumerate(history.history.get("val_loss", [])):
                    mlflow.log_metric("val_loss_epoch", float(val_loss), step=i)

                for i, train_acc in enumerate(history.history.get("accuracy", [])):
                    mlflow.log_metric("train_accuracy_epoch", float(train_acc), step=i)

                for i, val_acc in enumerate(history.history.get("val_accuracy", [])):
                    mlflow.log_metric("val_accuracy_epoch", float(val_acc), step=i)

                model.save(self.model_path)
                logging.info("Model saved to %s", self.model_path)

                mlflow.log_artifact(self.metrics_path, artifact_path="metrics")

                self._save_confusion_matrix(y_test, y_pred)
                mlflow.log_artifact(self.cm_path, artifact_path="plots")

                report = classification_report(y_test, y_pred, output_dict=False)
                report_path = os.path.join(self.artifact_dir, "classification_report.txt")
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path, artifact_path="reports")

                mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path="model",
                )

                logging.info("MLflow logging completed successfully")

            return metrics

        except Exception as e:
            logging.exception("Error during model training")
            raise CustomException(e, sys)