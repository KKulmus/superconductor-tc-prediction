"""src/models.py"""

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all ML models (Strategy Pattern)."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def _build_model(self):
        """Create the underlying sklearn model. Must be implemented by subclasses."""
        pass
    
    def train(self, X_train, y_train):
        """
        Train the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        if self.model is None:
        	self.model = self._build_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction!")
      
        return self.model.predict(X) 
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate predictions using RMSE and R².
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with 'rmse' and 'r2'
        """
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {"rmse": rmse, "r2": r2}


class LinearRegressionModel(BaseModel):
    """Concrete model using Linear Regression."""
    
    def __init__(self):
        super().__init__(name="Linear Regression")
    
    def _build_model(self):
        return LinearRegression()


class RandomForestModel(BaseModel):
    """Concrete model using Random Forest Regressor."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 16, random_state: int = 42):
        super().__init__(name="Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
    
    def _build_model(self):
        return RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        
        
        
        
        
        
        
        
