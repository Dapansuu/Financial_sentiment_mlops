import os
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import re 

class DataPreprocessing:
    def __init__(self):
        self.data_path = os.path.join("dataset", "raw", "data.csv")

    def clean_text(self, text: str) -> str:
        try:
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+", " ", text)
            text = re.sub(r"\S+@\S+", " ", text)
            text = re.sub(r"<.*?>", " ", text)
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        except Exception as e:
            logging.exception("Error during text cleaning")
            raise CustomException(e, sys)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting preprocessing step")

            df = df.copy()

            df = df.dropna(subset=["sentence", "sentiment"])
            logging.info(f"Shape after dropping nulls: {df.shape}")

            df["sentence"] = df["sentence"].astype(str).apply(self.clean_text)
            df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

            sentiment_mapping = {
                "neutral": 0,
                "positive": 1,
                "negative": 2
            }

            df = df[df["sentiment"].isin(sentiment_mapping.keys())]
            df["sentiment"] = df["sentiment"].map(sentiment_mapping)

            df = df.drop_duplicates(subset=["sentence", "sentiment"]).reset_index(drop=True)

            logging.info(f"Shape after preprocessing: {df.shape}")
            logging.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

            return df

        except Exception as e:
            logging.exception("Preprocessing failed")
            raise CustomException(e, sys)