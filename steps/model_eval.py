from src.Evaluation import R2_Score, MSE, RMSE
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
        
        r2_score = R2_Score()
        r2_score.evaluate(y_true, y_pred)
        
        mse = MSE()
        mse.evaluate(y_true, y_pred)
        
        rmse = RMSE()
        rmse.evaluate(y_true, y_pred)
        
        logging.info("Evaluation complete!")
        logging.info("R2 Score: {}".format(r2_score))
        logging.info("MSE: {}".format(mse))
        logging.info("RMSE: {}".format(rmse))
        
    except Exception as e:
        logging.error("Error while evaluating model: {}".format(e))
        raise e
    