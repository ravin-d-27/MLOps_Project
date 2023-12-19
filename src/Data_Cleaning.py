import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataStrategy(ABC):
    """Abstract class for data ingestion strategy."""
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    @abstractmethod
    def handle_data(self):
        """Reads the csv file and returns a Pandas DataFrame."""
        pass
    

class DataPreprocessing(DataStrategy):
    """Concrete class for data Preprocessing strategy."""
    
    def handle_data(self, data: pd.DataFrame)->pd.DataFrame:
        
        """This method preprocesses the data.
        Args:
            data: Pandas DataFrame.
        """
        
        try:
            logging.info("Starting Data Preprocessing")
            label_encoder = LabelEncoder()
            data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Embarked'], axis=1)
            data['Sex'] = label_encoder.fit_transform(data['Sex'])
            data = data.drop(columns=['Sex'])
            data["Age"].fillna(data["Age"].mean(), inplace=True)
            data["Fare"].fillna(data["Fare"].mean(), inplace=True)
            logging.info("Data Preprocessing is successful.")
            return data            
        except Exception as e:
            logging.error("Data Preprocessing is failed.")
            raise e

class DataSplitting_XandY(DataStrategy):
    """Concrete class for data Split for X and y strategy."""
    
    def handle_data(self, data: pd.DataFrame)->Tuple[pd.DataFrame, pd.Series]:
        """This method splits the data into X and y.
        
        Args:
            data: Pandas DataFrame.
        Returns:
            X: Pandas DataFrame.
            y: Pandas Series.
        """
        try:
            logging.info("Starting Data Splitting")
            X = data.drop(['Survived'], axis=1)
            y = data['Survived']
            logging.info("Data Splitting is successful.")
            return X, y            
        except Exception as e:
            logging.error("Data Splitting is failed.")
            raise e
        

class Data_train_test(DataStrategy):
    """Concrete class for data train test split strategy."""
    
    def handle_data(self, X: pd.DataFrame, y:pd.Series)->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        """
        This method handles the data by splitting it into training and testing sets.
        
        Parameters:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.
            
        Returns:
            Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]: A tuple containing the training and testing sets of X and y.
        """
        
        try:
            logging.info("Starting Data Splitting for training and testing")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Data Splitting for training and testing is successful.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("Data Splitting for training and testing is failed.")
            raise e