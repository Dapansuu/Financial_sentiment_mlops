import os
import sys
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.exception import CustomException
from src.utils.logger import logging


@dataclass
class DataTransformationConfig:
    raw_data_path: str = os.path.join("dataset", "raw", "raw.csv")
    
    interim_dir: str = os.path.join("dataset", "interim")
    
    train_path: str = os.path.join("dataset", "interim", "train.csv")
    test_path: str = os.path.join("dataset", "interim", "test.csv")
    val_path: str = os.path.join("dataset", "interim", "val.csv")
    
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


class DataPreprocessing:
    def __init__(self):
        try:
            self.config = DataTransformationConfig()
            self.sentiment_mapping = {
                "neutral": 0,
                "positive": 1,
                "negative": 2
            }
            logging.info("DataPreprocessing initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.config.raw_data_path):
                raise FileNotFoundError(f"Raw data file not found at: {self.config.raw_data_path}")

            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Raw data loaded successfully with shape: {df.shape}")
            return df

        except Exception as e:
            logging.exception("Failed to load raw data")
            raise CustomException(e, sys)

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

            logging.info(f"Original shape: {df.shape}")

            required_columns = ["sentence", "sentiment"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            df = df.dropna(subset=["sentence", "sentiment"])
            logging.info(f"Shape after dropping nulls: {df.shape}")

            df["sentence"] = df["sentence"].astype(str).apply(self.clean_text)
            df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

            df = df[df["sentence"].str.strip() != ""]
            logging.info(f"Shape after removing empty cleaned sentences: {df.shape}")

            df = df[df["sentiment"].isin(self.sentiment_mapping.keys())]
            logging.info(f"Shape after filtering valid sentiments: {df.shape}")

            label_counts = df.groupby("sentence")["sentiment"].nunique()
            conflicting_sentences = label_counts[label_counts > 1].index

            if len(conflicting_sentences) > 0:
                logging.warning(f"Removing {len(conflicting_sentences)} conflicting sentences")
                df = df[~df["sentence"].isin(conflicting_sentences)]

            df["sentiment"] = df["sentiment"].map(self.sentiment_mapping)

            df = df.drop_duplicates(subset=["sentence", "sentiment"]).reset_index(drop=True)

            logging.info(f"Final shape after preprocessing: {df.shape}")
            logging.info(f"Encoded sentiment distribution:\n{df['sentiment'].value_counts()}")

            return df

        except Exception as e:
            logging.exception("Preprocessing failed")
            raise CustomException(e, sys)

    def split_data(self, df: pd.DataFrame):
        try:
            train_df, temp_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df["sentiment"]
            )

            val_ratio_in_temp = self.config.val_size / self.config.test_size

            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_ratio_in_temp,
                random_state=self.config.random_state,
                stratify=temp_df["sentiment"]
            )

            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")
            logging.info(f"Val shape: {val_df.shape}")

            return train_df, test_df, val_df

        except Exception as e:
            logging.exception("Data splitting failed")
            raise CustomException(e, sys)

    def save_data(self, train_df, test_df, val_df):
        try:
            os.makedirs(self.config.interim_dir, exist_ok=True)

            train_df.to_csv(self.config.train_path, index=False)
            test_df.to_csv(self.config.test_path, index=False)
            val_df.to_csv(self.config.val_path, index=False)

            logging.info("Train, test, val saved to interim folder")
            logging.info(f"Train path: {self.config.train_path}")
            logging.info(f"Test path: {self.config.test_path}")
            logging.info(f"Val path: {self.config.val_path}")

            return (
                self.config.train_path,
                self.config.test_path,
                self.config.val_path
            )

        except Exception as e:
            logging.exception("Failed while saving processed datasets")
            raise CustomException(e, sys)

    def data_preprocessing(self):
        try:
            df = self.load_data()
            processed_df = self.preprocess_data(df)
            train_df, test_df, val_df = self.split_data(processed_df)
            return self.save_data(train_df, test_df, val_df)

        except Exception as e:
            logging.exception("Data preprocessing stage failed")
            raise CustomException(e, sys)