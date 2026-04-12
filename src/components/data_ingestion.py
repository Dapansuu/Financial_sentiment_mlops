import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.exception import CustomException
from src.utils.logger import logging


@dataclass
class DataIngestionConfig:
    dataset_dir: str = "dataset"
    train_data_path: str = os.path.join("dataset", "train_df.csv")
    test_data_path: str = os.path.join("dataset", "test_df.csv")
    val_data_path: str = os.path.join("dataset", "val_df.csv")
    source_data_path: str = os.path.join("dataset", "data.csv")
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


class DataIngestion:
    def __init__(self):
        try:
            self.ingestion_config = DataIngestionConfig()
            logging.info("DataIngestion initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self) -> pd.DataFrame:
        try:
            source_path = self.ingestion_config.source_data_path
            logging.info(f"Loading dataset from: {source_path}")

            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Dataset file not found at: {source_path}")

            df = pd.read_csv(source_path)
            logging.info(f"Dataset loaded successfully with shape: {df.shape}")
            return df

        except Exception as e:
            logging.exception("Failed while loading dataset")
            raise CustomException(e, sys)

    def validate_columns(self, df: pd.DataFrame) -> None:
        try:
            required_columns = ["sentence", "sentiment"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            

            logging.info("Required columns validation passed")

        except Exception as e:
            logging.exception("Column validation failed")
            raise CustomException(e, sys)

    def save_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> None:
        try:
            os.makedirs(self.ingestion_config.dataset_dir, exist_ok=True)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            val_df.to_csv(self.ingestion_config.val_data_path, index=False)

            logging.info("Train, test, and validation files saved successfully")
            logging.info(f"Train path: {self.ingestion_config.train_data_path}")
            logging.info(f"Test path: {self.ingestion_config.test_data_path}")
            logging.info(f"Val path: {self.ingestion_config.val_data_path}")

        except Exception as e:
            logging.exception("Failed while saving split datasets")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            df = self.load_data()
            self.validate_columns(df)

            logging.info("Starting train/test/validation split")

            # First split: train = 80%, temp = 20%
            train_df, temp_df = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                stratify=df["sentiment"]
            )

            # Second split: temp(20%) -> val(10%) + test(10%)
            val_ratio_in_temp = self.ingestion_config.val_size / (
                self.ingestion_config.test_size
            )

            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_ratio_in_temp,
                random_state=self.ingestion_config.random_state,
                stratify=temp_df["sentiment"]
            )

            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")
            logging.info(f"Validation shape: {val_df.shape}")

            self.save_data(train_df, test_df, val_df)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.val_data_path,
            )

        except Exception as e:
            logging.exception("Data ingestion process failed")
            raise CustomException(e, sys)


def main():
    try:
        data_ingestion = DataIngestion()
        train_path, test_path, val_path = data_ingestion.initiate_data_ingestion()

        logging.info("Data ingestion completed successfully")
        logging.info(f"Train file saved at: {train_path}")
        logging.info(f"Test file saved at: {test_path}")
        logging.info(f"Validation file saved at: {val_path}")

    except Exception as e:
        logging.exception("Error in main function")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()