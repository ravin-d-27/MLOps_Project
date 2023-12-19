import logging
from src.Model_Dev import Model, SupportVectorMachine
from zenml import step
import pandas as pd

from sklearn.base import ClassifierMixin

from sklearn.preprocessing import LabelEncoder

@step
def train_model(X_train:pd.DataFrame, y_train:pd.Series) -> ClassifierMixin:
    """Train the model."""
    try:
        logging.info("Starting Model Training")
        
        print(X_train, y_train)
        model = SupportVectorMachine()
        new_model = model.train(X_train, y_train)
        logging.info("Model Training is successful.")
        return new_model
    except Exception as e:
        logging.error("Model Training is failed.")
        raise e
    