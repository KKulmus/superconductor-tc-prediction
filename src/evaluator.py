"""src/evaluator.py"""

import matplotlib.pyplot as plt
import numpy as np


class ModelEvaluator:
    """Visualizes model performance and results."""
    
    def __init__(self, model_name: str = "Model"):
        self.model_name = model_name
    
    def plot_predicted_vs_observed(self, y_true, y_pred, ax=None):
        """
        Scatter plot of predicted vs. observed values with a diagonal reference line.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            ax: Optional matplotlib axis (for subplots)
            
        Returns:
            matplotlib Axes object
        """
       
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_title(f"{self.model_name}: Predicted vs. Observed")
        ax.set_xlabel("Observed Tc")
        ax.set_ylabel("Predicted Tc")
        
        return ax
    
    def plot_residuals(self, y_true, y_pred, ax=None):
        """
        Scatter plot of residuals vs. predicted values.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            ax: Optional matplotlib axis
            
        Returns:
            matplotlib Axes object
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.3, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title(f"{self.model_name}: Residuals")
        ax.set_xlabel("Predicted Tc")
        ax.set_ylabel("Residuals")
        
        return ax
        
        
        
        
        
        
        
        
        
