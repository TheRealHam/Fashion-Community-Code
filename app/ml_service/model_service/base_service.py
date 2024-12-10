import torch
from typing import Callable, Dict, Any  # noqa: F401


class ModelManager:
    """Manages dynamic model loading, prediction, updating, and evaluation."""

    def __init__(self):
        self.models = {}  # Registry for models
        self.preprocessors = {}  # Preprocessors for each model
        self.update_functions = {}  # Update logic for each model
        self.evaluation_functions = {}  # Evaluation logic for each model

    def register_model(
        self, 
        name: str, 
        model: torch.nn.Module, 
        model_path: str, 
        preprocess: Callable, 
        update_fn: Callable, 
        evaluate_fn: Callable
    ):
        """
        Register a new model with its associated logic.
        Args:
            name (str): Unique name of the model.
            model (torch.nn.Module): PyTorch model instance.
            model_path (str): Path to the saved model weights.
            preprocess (Callable): Function for preprocessing input data.
            update_fn (Callable): Function to update the model.
            evaluate_fn (Callable): Function to evaluate the model.
        """
        # Load the model weights and set to evaluation mode
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Register components
        self.models[name] = model
        self.preprocessors[name] = preprocess
        self.update_functions[name] = update_fn
        self.evaluation_functions[name] = evaluate_fn

        print(f"Model '{name}' registered successfully.")

    def predict(self, name: str, input_data: Any) -> Any:
        """
        Make a prediction using a registered model.
        Args:
            name (str): Name of the model.
            input_data (Any): Raw input data.

        Returns:
            Any: Prediction result.
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' is not registered.")
        
        # Preprocess input data
        processed_data = self.preprocessors[name](input_data)

        # Perform prediction
        with torch.no_grad():
            return self.models[name](processed_data)

    def update_model(self, name: str, *args, **kwargs):
        """
        Update a registered model.
        Args:
            name (str): Name of the model to update.
        """
        if name not in self.update_functions:
            raise ValueError(f"Update function for model '{name}' is not registered.")
        
        self.update_functions[name](*args, **kwargs)

    def evaluate_model(self, name: str, *args, **kwargs) -> Any:
        """
        Evaluate a registered model.
        Args:
            name (str): Name of the model to evaluate.

        Returns:
            Any: Evaluation results.
        """
        if name not in self.evaluation_functions:
            raise ValueError(f"Evaluation function for model '{name}' is not registered.")
        
        return self.evaluation_functions[name](*args, **kwargs)
