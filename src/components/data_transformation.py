import os
import sys
import pandas as pd
import tensorflow as tf

from src.utils.exception import CustomException
from src.utils.logger import logging


class DataTransformation:

    def __init__(self):
        self.data_path = os.path.join("dataset", "train.csv")

    def data_transformation(self):

        try:
            df = pd.read_csv(self.data_path)

            X = df["sentence"]
            y = df["sentiment"]
            
            max_tokens = 12000
            sequence_length = 40
            
            text_vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=max_tokens,
                output_mode="int",
                output_sequence_length=sequence_length,
                standardize="lower_and_strip_punctuation"  
            )

            # adapt only on training text
            text_vectorizer.adapt(X)

            logging.info("Data transformation completed")

            return X, y, text_vectorizer

        except Exception as e:
            raise CustomException(e, sys)