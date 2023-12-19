from src.Evaluation import accuracy, MSE, RMSE
import logging
from zenml import step
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

@step
def model_eval(X_test:pd.DataFrame, y_true: pd.DataFrame, model: ClassifierMixin)->None:
    """This step evaluates the performance of a model."""
    
    try:
        logging.info("Evaluating model...")
        
        y_pred = model.predict(X_test)
        
        acc = accuracy()
        acc.evaluate(y_true, y_pred)
        
        mse = MSE()
        mse.evaluate(y_true, y_pred)
        
        rmse = RMSE()
        rmse.evaluate(y_true, y_pred)
        
        logging.info("Evaluation complete!")
        logging.info("Accuracy Score: {}".format(acc))
        logging.info("MSE: {}".format(mse))
        logging.info("RMSE: {}".format(rmse))
        
    except Exception as e:
        logging.error("Error while evaluating model: {}".format(e))
        raise e
    