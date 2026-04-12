from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def start_training(self):

        ingestion = DataIngestion()
        data_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        X, y, vectorizer = transformation.initiate_data_transformation(data_path)

        trainer = ModelTrainer()
        accuracy = trainer.initiate_model_trainer(X, y, vectorizer)

        print("Training Completed")
        print("Accuracy:", accuracy)