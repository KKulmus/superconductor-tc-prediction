"""src/data_loader.py"""

# ABC [Abstract Base Class] guarantees that all data_loader classes
# need to implement certain methods.

from abc import ABC, abstractmethod
# ABC is the base class which other classes inherit.
# It is abstract, hence no object can be created direct from it
# abstractmethod is a decorator: Each subclass must implement this method, otherwise an error is thrown
import pandas as pd
# standart for manipulating data
from pathlib import Path
# modern modul, replaces os.path. Is inherently OOP
from typing import Optional
# modul for type hints

class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""

    def __init__(self, file_path:str):
        self.file_path = Path(file_path)
        self.data: Optional[pd.DataFrame] = None

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and return the dataset. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def validate(seld, df: pd.DataFrame) -> bool:
        """Validate dataset integrity. Must be implemented by subclasses."""
        pass
    
class SuperconductorDataLoader(BaseDataLoader):
    """Concrete loader for the superconductivity dataset."""
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
    
    def load(self) -> pd.DataFrame:
        """
        Load CSV file from path.
        
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If essential columns are missing
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        # Try loading as CSV (handles various separators)
        try:
            self.data = pd.read_csv(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")
        
        print(f"Loaded {len(self.data)} rows from {self.file_path.name}")
        return self.data
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate that required columns exist and no critical NaN values.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        # Check for target column
        if 'critical_temp' not in df.columns:
            print("Missing column: 'critical_temp'")
            return False
        
        # Check for material/formula column
        if 'material' not in df.columns and len(df.columns) > 50:
            # Many features might imply no explicit formula column
            pass  # OK if it's feature-only format
        
        # Check for NaN in target
        nan_count = df['critical_temp'].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values in target variable")
            return False
        
        print("Data validation passed")
        return True


    def get_data_source_urls() -> dict:
        """Return URLs for downloading the datasets."""
        return {
            "train": "https://archive.ics.uci.edu/ml/machine-learning-databases/00457/train.csv",
            "materials": "https://archive.ics.uci.edu/ml/machine-learning-databases/00457/unique_m.csv",
            "repository": "https://archive.ics.uci.edu/ml/datasets/superconductivty+data"
            }
