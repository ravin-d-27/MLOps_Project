from src.Data_Cleaning import DataStrategy, Data_train_test
from zenml import step
import logging
import pandas as pd
from typing import Tuple

@step
def train_and_test_split(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    """
    Data cleaning step.
    """
    try:
        logging.info("Starting Data Splitting into train and test")
        data_preprocessing = Data_train_test()
        X_train, X_test, y_train, y_test = data_preprocessing.handle_data(X,y)
        logging.info("Data Splitting of train and test is successful.")
        return X_train, X_test, y_train, y_test          
    except Exception as e:
        logging.error("Data Splitting of train and test is failed.")
        raise e
