import os
import sys
from dataclasses import dataclass
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logging


@dataclass
class DataIngestionConfig:
    source_data_path: str = os.path.join("dataset", "data.csv")
    raw_data_dir: str = os.path.join("dataset", "raw")
    raw_data_path: str = os.path.join("dataset", "raw", "raw.csv")


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

            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Dataset file not found at: {source_path}")

            df = pd.read_csv(source_path)

            if df.empty:
                raise ValueError("Input dataset is empty")

            logging.info(f"Dataset loaded successfully from {source_path} with shape {df.shape}")
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

            logging.info("Column validation passed")

        except Exception as e:
            logging.exception("Column validation failed")
            raise CustomException(e, sys)

    def save_raw_data(self, df: pd.DataFrame) -> str:
        try:
            os.makedirs(self.ingestion_config.raw_data_dir, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw dataset saved at {self.ingestion_config.raw_data_path}")
            return self.ingestion_config.raw_data_path

        except Exception as e:
            logging.exception("Failed while saving raw dataset")
            raise CustomException(e, sys)

    def data_ingestion(self) -> str:
        try:
            df = self.load_data()
            self.validate_columns(df)
            raw_path = self.save_raw_data(df)
            logging.info("Data ingestion completed successfully")
            return raw_path

        except Exception as e:
            logging.exception("Data ingestion stage failed")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        data_ingestion.data_ingestion()
    except Exception as e:
        logging.exception("Error in main execution")
        raise CustomException(e, sys)