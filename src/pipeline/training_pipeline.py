import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.exception import CustomException
from src.utils.logger import logging


class TrainingPipeline:
    def __init__(self):
        try:
            logging.info("TrainingPipeline initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # Stage 1: Data Ingestion
            logging.info("Starting data ingestion stage")
            ingestion = DataIngestion()
            raw_data_path = ingestion.data_ingestion()
            logging.info(f"Data ingestion completed. Raw data saved at: {raw_data_path}")

            # Stage 2: Data Preprocessing
            logging.info("Starting data preprocessing stage")
            preprocessing = DataPreprocessing()
            train_path, test_path, val_path = preprocessing.data_preprocessing()
            logging.info(
                "Data preprocessing completed. "
                f"Train: {train_path}, Test: {test_path}, Val: {val_path}"
            )

            # Stage 3: Data Transformation / Vectorization
            logging.info("Starting data transformation stage")
            transformation = DataTransformation()
            transformed_train, transformed_test, transformed_val, vectorizer_path = (
                transformation.transform_data()
            )
            logging.info(
                "Data transformation completed. "
                f"Train: {transformed_train}, Test: {transformed_test}, "
                f"Val: {transformed_val}, Vectorizer: {vectorizer_path}"
            )

            # Stage 4: Model Training
            logging.info("Starting model trainer stage")
            trainer = ModelTrainer()
            metrics = trainer.model_trainer()
            logging.info(f"Model training completed successfully. Metrics: {metrics}")

            logging.info("Training pipeline completed successfully")
            return metrics

        except Exception as e:
            logging.exception("Training pipeline failed")
            raise CustomException(e, sys)


def main():
    try:
        pipeline = TrainingPipeline()
        metrics = pipeline.run_pipeline()
        print("Pipeline completed successfully")
        print("Final metrics:")
        print(metrics)

    except Exception as e:
        logging.exception("Error in training_pipeline main")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()