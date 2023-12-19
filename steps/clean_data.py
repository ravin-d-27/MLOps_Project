from src.Data_Cleaning import DataPreprocessing, DataStrategy
from zenml import step
import logging
import pandas as pd

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning step.
    """
    try:
        logging.info("Starting Data Cleaning")
        data_preprocessing = DataPreprocessing(DataStrategy)
        df = data_preprocessing.handle_data(df)
        logging.info("Data Cleaning is successful.")
        return df            
    except Exception as e:
        logging.error("Data Cleaning is failed.")
        raise e
