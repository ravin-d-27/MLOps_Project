from src.Data_Cleaning import DataSplitting_XandY, DataStrategy
from zenml import step
import logging
import pandas as pd
from typing import Tuple

@step
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Data cleaning step.
    """
    try:
        logging.info("Starting Data Splitting into X and Y")
        data_preprocessing = DataSplitting_XandY(DataStrategy)
        X,y = data_preprocessing.handle_data(df)
        logging.info("Data Splitting os X and y is successful.")
        return X,y          
    except Exception as e:
        logging.error("Data Splitting of X and y is failed.")
        raise e
