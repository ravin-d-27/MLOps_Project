import logging
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from abc import ABC, abstractmethod


class Evaluation(ABC):
    """This class represents an evaluation object."""
    
    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """This method evaluates the performance of a model."""
        pass
    

class accuracy(Evaluation):
    """This class represents a R2_Score object."""
    
    def evaluate(self, y_true:np.ndarray, y_pred:np.ndarray):
        """This method evaluates the performance of a model."""
        
        try:
            logging.info("Evaluating R2 Score of the model...")
            score = accuracy_score(y_true, y_pred)
            logging.info("Accuracy Score: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error while evaluating R2 Score of the model: {}".format(e))
            raise e
        
class MSE(Evaluation):
    """This class represents a MSE object."""
    
    def evaluate(self, y_true:np.ndarray, y_pred:np.ndarray):
        """This method evaluates the performance of a model."""
        
        try:
            logging.info("Evaluating MSE of the model...")
            score = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error while evaluating MSE of the model: {}".format(e))
            raise e

class RMSE(Evaluation):
    """This class represents a RMSE object."""
    
    def evaluate(self, y_true:np.ndarray, y_pred:np.ndarray):
        """This method evaluates the performance of a model."""
        
        try:
            logging.info("Evaluating RMSE of the model...")
            score = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error while evaluating RMSE of the model: {}".format(e))
            raise e

    
    
