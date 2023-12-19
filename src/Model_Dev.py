import logging
from abc import ABC, abstractmethod
from sklearn.svm import SVC

class Model(ABC):
    """Abstract class for model strategy."""
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model."""
        pass
    
class SupportVectorMachine(ABC):
    
    """Concrete class for Support Vector Machine model strategy."""
    
    def train(self, X_train, y_train):
        
        """Train the model."""
        
        try:
            logging.info("Starting Model Training")
            self.model = SVC()
            self.model.fit(X_train, y_train)
            logging.info("Model Training is successful.")
            return self.model
        except Exception as e:
            logging.error("Model Training is failed.")
            raise e