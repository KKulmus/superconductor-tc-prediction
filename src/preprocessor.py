"""src/preprocessor.py"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handles feature-target separation, train-test split, and scaling."""
    
    def __init__(self, test_size: float = 0.33, random_state: int = 42):
        """
        Initialize preprocessor with split parameters.
        
        Args:
            test_size: Fraction of data reserved for testing (default: 0.33)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def separate_features_target(self, df: pd.DataFrame, target_column: str = "critical_temp"):
        """
        Split DataFrame into features (X) and target (y).
        
        Args:
            df: The full DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X, y) where X is a DataFrame and y is a Series
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        assert isinstance(X, pd.DataFrame), "X must be type Dataframe"
        assert isinstance(y, pd.Series), "Target must be of type Series"
        return X,y
    
    def split_data(self, X, y):
        """
        Split data into training and test sets.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Fit scaler on training data, then transform both train and test.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple: (X_train_scaled, X_test_scaled)
        """
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def run(self, df: pd.DataFrame, target_column: str = "critical_temp"):
        """
        Run the full preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            target_column: Name of target column
            
        Returns:
            Dictionary with keys: X_train, X_test, y_train, y_test
        """
        assert isinstance(df, pd.DataFrame), f"df must be of type Dataframe, but is {type(df)}"
        assert isinstance(target_column, str), f"Target Column must be of type string, but is {type(target_column)}"
        
        X, y = self.separate_features_target(df, target_column)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        preprocessed_data_dictionary = {
    		"X_train": X_train_scaled,
    		"X_test": X_test_scaled,
    		"y_train": y_train,
    		"y_test": y_test
		}
        return preprocessed_data_dictionary
        
        
        
        
        
        
        
