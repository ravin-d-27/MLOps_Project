import logging
import pandas as pd
from src.Data_Cleaning import DataPreprocessing, DataSplitting_XandY, Data_train_test


def get_data_for_test():
    try:
        df = pd.read_csv('/home/ravind27/Desktop/My_Projects_and_Codes/MLOps_Project/data/Titanic_Dataset.csv')
        df = df.sample(n=100)
        data_preprocessing = DataPreprocessing()
        df = data_preprocessing.handle_data(df)
        result = df.to_json(orient='split')
        return result
    except Exception as e:
        logging.error(e)
        raise e
    
