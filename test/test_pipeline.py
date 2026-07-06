"""tests/test_pipeline.py"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from src.data_loader import SuperconductorDataLoader, BaseDataLoader
from src.preprocessor import DataPreprocessor
from src.models import LinearRegressionModel, RandomForestModel


class TestDataLoader:
    """Tests for the SuperconductorDataLoader."""
    
    def test_load_returns_dataframe(self):
        """Loader should return a pandas DataFrame."""
        loader = SuperconductorDataLoader("data/raw/train.csv")
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
    
    def test_load_correct_shape(self):
        """Loaded data should have 21,263 rows and 82 columns."""
        loader = SuperconductorDataLoader("data/raw/train.csv")
        df = loader.load()
        assert df.shape == (21263, 82)
    
    def test_validate_passes_on_good_data(self):
        """Validation should pass on the real dataset."""
        loader = SuperconductorDataLoader("data/raw/train.csv")
        df = loader.load()
        assert loader.validate(df) is True
    
    def test_load_raises_on_missing_file(self):
        """Loader should raise FileNotFoundError for non-existent file."""
        loader = SuperconductorDataLoader("data/raw/nonsense.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()


class TestDataPreprocessor:
    """Tests for the DataPreprocessor."""
    
    def setup_method(self):
        """Create a small dummy dataset for each test."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            np.random.rand(100, 5),
            columns=["f1", "f2", "f3", "f4", "critical_temp"]
        )
        self.preprocessor = DataPreprocessor(test_size=0.33, random_state=42)
    
    def test_separate_features_target(self):
        """Should return X with 4 columns and y with 1 column."""
        X, y = self.preprocessor.separate_features_target(self.df)
        assert X.shape == (100, 4)
        assert y.shape == (100,)
        assert "critical_temp" not in X.columns
    
    def test_split_data_shapes(self):
        """Train + Test should equal total samples."""
        X, y = self.preprocessor.separate_features_target(self.df)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        assert X_train.shape[0] + X_test.shape[0] == 100
        assert y_train.shape[0] + y_test.shape[0] == 100
    
    def test_scale_features_no_nan(self):
        """Scaled features should not contain NaN values."""
        X, y = self.preprocessor.separate_features_target(self.df)
        X_train, X_test, _, _ = self.preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        assert not np.isnan(X_train_scaled).any()
        assert not np.isnan(X_test_scaled).any()
    
    def test_run_returns_dictionary(self):
        """run() should return a dict with the correct keys."""
        result = self.preprocessor.run(self.df)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"X_train", "X_test", "y_train", "y_test"}


class TestModels:
    """Tests for the model classes (Strategy Pattern)."""
    
    def setup_method(self):
        """Create small dummy training data."""
        np.random.seed(42)
        self.X_train = np.random.rand(50, 5)
        self.y_train = np.random.rand(50)
        self.X_test = np.random.rand(10, 5)
    
    def test_linear_regression_train_predict(self):
        """LinearRegressionModel should train and predict."""
        model = LinearRegressionModel()
        model.train(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        assert preds.shape == (10,)
        assert model.is_trained is True
    
    def test_random_forest_train_predict(self):
        """RandomForestModel should train and predict."""
        model = RandomForestModel()
        model.train(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        assert preds.shape == (10,)
        assert model.is_trained is True
    
    def test_predict_without_training_raises_error(self):
        """Calling predict() before train() should raise RuntimeError."""
        model = LinearRegressionModel()
        with pytest.raises(RuntimeError):
            model.predict(self.X_test)
    
    def test_evaluate_returns_metrics(self):
        """evaluate() should return a dict with 'rmse' and 'r2'."""
        model = LinearRegressionModel()
        model.train(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        metrics = model.evaluate(self.y_train[:10], preds)
        assert "rmse" in metrics
        assert "r2" in metrics
        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["r2"], float)
