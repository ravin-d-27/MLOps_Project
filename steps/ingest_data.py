import pandas as pd
from zenml import step
import logging


class IngestData:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        
    def get_data(self):
        """Reads the csv file and returns a Pandas DataFrame."""
        dataframe = pd.read_csv(self.data_path)
        return dataframe
    
@step
def run(data_path: str) -> pd.DataFrame:
    """Ingest data step.
    Args:
        data_path: Path to the data.
    Returns:
        df: Pandas DataFrame.
    """
    try:
        logging.info("Starting Data ingestion.")
        ingest = IngestData(data_path)
        df = ingest.get_data()
        logging.info("Data ingestion is successful.")
        return df
    except Exception as e:
        logging.error("Data ingestion is failed.")
        raise e


if __name__ == "__main__":
    data_path = 'data/Titanic.csv'
    result_df = run(data_path)
    print(result_df.head())

