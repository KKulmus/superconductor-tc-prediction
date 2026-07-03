"""src/pipeline.py"""

from src.data_loader import SuperconductorDataLoader
from src.preprocessor import DataPreprocessor
from src.evaluator import ModelEvaluator


class Pipeline:
    """Orchestrates the full ML workflow: load → preprocess → train → evaluate."""
    
    def __init__(self, model, data_path: str, target_column: str = "critical_temp"):
        """
        Initialize pipeline with all components via composition.
        
        Args:
            model: A BaseModel instance (e.g., LinearRegressionModel or RandomForestModel)
            data_path: Path to the CSV file
            target_column: Name of the target column
        """
        self.model = model
        self.data_path = data_path
        self.target_column = target_column
        
        # Composition: Pipeline "owns" these objects
        self.loader = SuperconductorDataLoader(data_path)
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator(model_name=model.name)
        
        # placeholder for results
        self.data = None
        self.processed_data = None
        self.metrics = None
        self.y_test = None
        self.y_pred = None
    
    def load_data(self):
        """
        Load and validate data using the DataLoader.
        """
        
        df = self.loader.load()
        if self.loader.validate(df):
            self.data = df
    
    def preprocess(self):
        """
        Run preprocessing on loaded data.
        """
       
        if self.data is None:
            raise ValueError("No data loaded!")
        self.processed_data = self.preprocessor.run(self.data, self.target_column)
        
    
    def train_model(self):
        """
        Train the model on preprocessed training data.
        """
        X_train = self.processed_data["X_train"]
        y_train = self.processed_data["y_train"]
        self.model.train(X_train, y_train)
        
    
    def evaluate_model(self):
        """
        Make predictions on test set and evaluate.
        """
    
        X_test = self.processed_data["X_test"]
        self.y_test = self.processed_data["y_test"]
        self.y_pred = self.model.predict(X_test)
        self.metrics = self.model.evaluate(self.y_test, self.y_pred)
    
    def run(self):
        """
        Execute the full pipeline: load → preprocess → train → evaluate.
        """
        self.load_data()
        self.preprocess()
        self.train_model()
        self.evaluate_model()
    
    def plot_results(self):
        """
        Show predicted-vs-observed and residual plots side by side.
        """
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        self.evaluator.plot_predicted_vs_observed(self.y_test, self.y_pred, ax=axes[0])
        self.evaluator.plot_residuals(self.y_test, self.y_pred, ax=axes[1])
        plt.tight_layout()
        plt.show()
