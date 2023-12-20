from src.Evaluation import accuracy, MSE, RMSE
import logging
from zenml import step
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_eval(X_test:pd.DataFrame, y_true: pd.DataFrame, model: ClassifierMixin)->None:
    """This step evaluates the performance of a model.
        Args:
            X_test: Pandas DataFrame.
            y_true: Pandas DataFrame.
            model: Trained model.
        Returns:
            None
    """
    
    try:
        logging.info("Evaluating model...")
        
        y_pred = model.predict(X_test)
        
        acc = accuracy()
        accuracy_score = acc.evaluate(y_true, y_pred)
        mlflow.log_metric("accuracy", accuracy_score)
        
        mse = MSE()
        MSE_score = mse.evaluate(y_true, y_pred)
        mlflow.log_metric("mse", MSE_score)
        
        rmse = RMSE()
        RMSE_score = rmse.evaluate(y_true, y_pred)
        mlflow.log_metric("rmse", RMSE_score)
        
        logging.info("Evaluation complete!")
        logging.info("Accuracy Score: {}".format(acc))
        logging.info("MSE: {}".format(mse))
        logging.info("RMSE: {}".format(rmse))
        
    except Exception as e:
        logging.error("Error while evaluating model: {}".format(e))
        raise e
    