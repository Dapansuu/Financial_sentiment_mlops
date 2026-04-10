import os
import sys
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.exception import CustomException
from src.utils.logger import logging


class DataIngestion:
    def __init__(self):
        self.data_path = os.path.join("dataset", "data.csv")
        self.df = None

        try:
            logging.info("Initializing DataIngestion class")
            logging.info(f"Reading dataset from: {self.data_path}")

            self.df = pd.read_csv(self.data_path)

            logging.info("Dataset loaded successfully")
            logging.info(f"Dataset shape: {self.df.shape}")

        except Exception as e:
            logging.exception("Error initializing DataIngestion")
            raise CustomException(e, sys)

    def clean_text(self, text: str) -> str:
        try:
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+", " ", text)
            text = re.sub(r"\S+@\S+", " ", text)
            text = re.sub(r"<.*?>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        except Exception as e:
            logging.exception("Error during text cleaning")
            raise CustomException(e, sys)

    def mapping(self):
        try:
            logging.info("Starting sentiment mapping")

            self.df["sentiment"] = self.df["sentiment"].map({
                "neutral": 0,
                "positive": 1,
                "negative": 2
            })

            logging.info("Sentiment mapping completed")
            logging.info(f"Sentiment distribution:\n{self.df['sentiment'].value_counts()}")

            return self.df

        except Exception as e:
            logging.exception("Error during sentiment mapping")
            raise CustomException(e, sys)

    def split_data(self, test_size=0.2, random_state=42):
        try:
            logging.info("Starting train-test split")

            X = self.df["sentence"]
            y = self.df["sentiment"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            logging.info("Train-test split completed")
            logging.info(f"Train size: {X_train.shape}")
            logging.info(f"Test size: {X_test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.exception("Error during train-test split")
            raise CustomException(e, sys)

    def data_ingest(self):
        logging.info("Starting full data ingestion pipeline")

        try:
            if self.df is None:
                raise ValueError(f"Dataset not loaded. Check file path: {self.data_path}")

            if "sentence" not in self.df.columns:
                raise ValueError("Column 'sentence' not found in dataset")

            if "sentiment" not in self.df.columns:
                raise ValueError("Column 'sentiment' not found in dataset")

            logging.info("Starting text cleaning")
            self.df["sentence"] = self.df["sentence"].apply(self.clean_text)
            logging.info("Text cleaning completed")

            self.mapping()

            logging.info("Data ingestion pipeline completed successfully")
            return self.df

        except Exception as e:
            logging.exception("Error in data ingestion pipeline")
            raise CustomException(e, sys)