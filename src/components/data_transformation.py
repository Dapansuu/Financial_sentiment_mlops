import os
import sys
from dataclasses import dataclass

import pandas as pd
import tensorflow as tf

from src.utils.exception import CustomException
from src.utils.logger import logging
import shutil

@dataclass
class DataTransformationConfig:
    interim_dir: str = os.path.join("dataset", "interim")
    train_path: str = os.path.join("dataset", "interim", "train.csv")
    test_path: str = os.path.join("dataset", "interim", "test.csv")
    val_path: str = os.path.join("dataset", "interim", "val.csv")
    max_tokens: int = 12000
    sequence_length: int = 40

class DataTransformation:
    def __init__(self):
        try:
            self.config = DataTransformationConfig()
            logging.info("DataTransformation initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):
        try:
            if not os.path.exists(self.config.train_path):
                raise FileNotFoundError(f"Train file not found at: {self.config.train_path}")
            if not os.path.exists(self.config.test_path):
                raise FileNotFoundError(f"Test file not found at: {self.config.test_path}")
            if not os.path.exists(self.config.val_path):
                raise FileNotFoundError(f"Validation file not found at: {self.config.val_path}")

            train_df = pd.read_csv(self.config.train_path)
            test_df = pd.read_csv(self.config.test_path)
            val_df = pd.read_csv(self.config.val_path)

            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")
            logging.info(f"Val shape: {val_df.shape}")

            return train_df, test_df, val_df

        except Exception as e:
            logging.exception("Failed while loading interim datasets")
            raise CustomException(e, sys)

    def split_features_labels(self, train_df, test_df, val_df):
        try:
            X_train = train_df["sentence"].astype(str)
            y_train = train_df["sentiment"]

            X_test = test_df["sentence"].astype(str)
            y_test = test_df["sentiment"]

            X_val = val_df["sentence"].astype(str)
            y_val = val_df["sentiment"]

            logging.info("Split features and labels successfully")

            return X_train, y_train, X_test, y_test, X_val, y_val

        except Exception as e:
            logging.exception("Failed while separating features and labels")
            raise CustomException(e, sys)

    def create_vectorizer(self):
        try:
            text_vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=self.config.max_tokens,
                output_mode="int",
                output_sequence_length=self.config.sequence_length,
                standardize="lower_and_strip_punctuation"
            )

            logging.info("Text vectorizer created successfully")
            return text_vectorizer

        except Exception as e:
            logging.exception("Failed while creating text vectorizer")
            raise CustomException(e, sys)
        

    def transform_data(self):
        try:
            train_df, test_df, val_df = self.load_data()

            X_train, y_train, X_test, y_test, X_val, y_val = self.split_features_labels(
                train_df, test_df, val_df
            )

            text_vectorizer = self.create_vectorizer()

            # Fit only on training data
            text_vectorizer.adapt(X_train)
            logging.info("Text vectorizer adapted on training data only")

            X_train_vec = text_vectorizer(X_train)
            X_test_vec = text_vectorizer(X_test)
            X_val_vec = text_vectorizer(X_val)

            os.makedirs("dataset/transformed", exist_ok=True)

            # Clean old outputs before saving
            dataset_paths = [
                "dataset/transformed/train",
                "dataset/transformed/test",
                "dataset/transformed/val",
            ]
            for path in dataset_paths:
                if os.path.exists(path):
                    shutil.rmtree(path)

            vectorizer_path = "dataset/transformed/vectorizer.keras"
            if os.path.exists(vectorizer_path):
                os.remove(vectorizer_path)

            # Save transformed data
            tf.data.Dataset.from_tensor_slices((X_train_vec, y_train)).save(
                "dataset/transformed/train"
            )
            tf.data.Dataset.from_tensor_slices((X_test_vec, y_test)).save(
                "dataset/transformed/test"
            )
            tf.data.Dataset.from_tensor_slices((X_val_vec, y_val)).save(
                "dataset/transformed/val"
            )

            # Save vectorizer
            vectorizer_model = tf.keras.Sequential([text_vectorizer])
            vectorizer_model.save(vectorizer_path)

            logging.info("Data transformation completed and saved successfully")

            return (
                "dataset/transformed/train",
                "dataset/transformed/test",
                "dataset/transformed/val",
                vectorizer_path
            )

        except Exception as e:
            logging.exception("Data transformation stage failed")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        transformation = DataTransformation()
        transformation.transform_data()
    except Exception as e:
        logging.exception("Error in main execution")
        raise CustomException(e, sys)