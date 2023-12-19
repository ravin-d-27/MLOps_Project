import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Tuple


class DataStrategy(ABC):
    """Abstract class for data ingestion strategy."""
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    @abstractmethod
    def handle_data(self):
        """Reads the csv file and returns a Pandas DataFrame."""
        pass
    

class DataPreprocessing(DataStrategy):
    """Concrete class for data ingestion strategy."""
    
    def handle_data(self, data: pd.DataFrame)->pd.DataFrame:
        
        try:
            logging.info("Starting Data Preprocessing")
            data = data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
            logging.info("Data Preprocessing is successful.")
            return data            
        except Exception as e:
            logging.error("Data Preprocessing is failed.")
            raise e

class DataSplitting_XandY(DataStrategy):
    """Concrete class for data ingestion strategy."""
    
    def handle_data(self, data: pd.DataFrame)->Tuple[pd.DataFrame, pd.Series]:
        
        try:
            logging.info("Starting Data Splitting")
            X = data.drop(['Survived'], axis=1)
            y = data['Survived']
            logging.info("Data Splitting is successful.")
            return X, y            
        except Exception as e:
            logging.error("Data Splitting is failed.")
            raise e