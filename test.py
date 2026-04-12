from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformation
from src.utils.logger import logging


def main():
    logging.info("Pipeline execution started")

    # Stage 1: ingestion
    ingestion = DataIngestion()
    raw_path = ingestion.data_ingestion()
    logging.info(f"Raw data saved at: {raw_path}")

    # Stage 2: preprocessing
    preprocessing = DataPreprocessing()
    train_path, test_path, val_path = preprocessing.data_preprocessing()
    logging.info(f"Preprocessed train path: {train_path}")
    logging.info(f"Preprocessed test path: {test_path}")
    logging.info(f"Preprocessed val path: {val_path}")

    # Stage 3: vectorization / transformation
    transformation = DataTransformation()
    transformed_train, transformed_test, transformed_val, vectorizer_path = transformation.transform_data()

    logging.info("Pipeline execution completed successfully")
    logging.info(f"Transformed train path: {transformed_train}")
    logging.info(f"Transformed test path: {transformed_test}")
    logging.info(f"Transformed val path: {transformed_val}")
    logging.info(f"Vectorizer path: {vectorizer_path}")

    print("Pipeline completed successfully")
    print("Artifacts created:")
    print(raw_path)
    print(train_path)
    print(test_path)
    print(val_path)
    print(transformed_train)
    print(transformed_test)
    print(transformed_val)
    print(vectorizer_path)

if __name__ == "__main__":
    main()