from zenml import pipeline

from steps.ingest_data import run
from steps.clean_data import clean_data
from steps.split_data import split_data
from steps.train_and_test_split import train_and_test_split
from steps.model_train import train_model
from steps.model_eval import model_eval

import pandas as pd


@pipeline
def training_pipeline(data_path: str):
    """
    Training pipeline.
    """
    df_artifact = run(data_path)
    df_cleaned = clean_data(df_artifact)
    X,y = split_data(df_cleaned)
    X_train, X_test, y_train, y_test = train_and_test_split(X,y)

    model = train_model(X_train, y_train)
    accuracy_score=model_eval(X_test, y_test, model)
    
    