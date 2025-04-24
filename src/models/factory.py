import torch
import os
import glob
from src.models.colontcn import ColonTCN

class ModelFactory:
    """
    Factory class for model creation, instantiation, and state loading based on a configuration.
    """

    def __init__(self, config):
        """
        Initialize ModelFactory using a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self, load_checkpoint=True):
        """
        Instantiates a model based on the configuration.

        Args:
            load_checkpoint (bool): Whether to load weights from a checkpoint.

        Returns:
            torch.nn.Module: An instance of the specified model.
        """
        model_type = self.config.get("model_type")
        model_class = self._get_model_class(model_type)

        # Extract parameters and initialize the model
        model_params = self._extract_model_params()
        model = model_class(**model_params)

        # Load model state if a model path is provided
        if "model_path" in self.config and load_checkpoint and self.config["model_path"]!="":
            model = self.load_checkpoint(model, self.config["model_path"])

        model.to(self.device)
        return model

    def load_checkpoint(self, model, checkpoint_path):
        """
        Loads a model from a specific checkpoint.

        Args:
            model (torch.nn.Module): Model instance.
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            torch.nn.Module: Model with loaded state.
        """
        # Handle wildcard paths (e.g., model_path contains multiple checkpoints)
        if "*" in checkpoint_path:
            checkpoint_files = sorted(glob.glob(checkpoint_path), key=os.path.getmtime, reverse=True)
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoints found matching pattern: {checkpoint_path}")
            checkpoint_path = checkpoint_files[0]  # Load the latest checkpoint

        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Handle missing or unexpected keys
        corrected_state_dict = self._correct_checkpoint_keys(checkpoint['model_state_dict'], model)

        try:
            model.load_state_dict(corrected_state_dict)
        except RuntimeError as e:
            print(f"RuntimeError during state_dict loading: {e}. Attempting to reshape weights...")
            self._correct_checkpoint_shape(model, corrected_state_dict)

        return model

    def _get_model_class(self, model_type):
        """
        Retrieves the model class based on the model type.

        Args:
            model_type (str): Type of model specified in the configuration.

        Returns:
            class: The model class corresponding to the model type.

        Raises:
            ValueError: If the model type is not supported.
        """
        model_classes = {"colontcn": ColonTCN}
        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_classes[model_type]

    def _extract_model_params(self):
        """
        Extracts model-specific parameters from the configuration.

        Returns:
            dict: Dictionary of parameters specific to the model.
        """
        return {
            "input_size": self.config["input_size"],
            "list_of_features_sizes": self.config["list_of_features_sizes"],
            "num_of_convs": self.config["num_of_convs"],
            "kernel_size": self.config["kernel_size"],
            "dropout": self.config["dropout"],
            "residual": self.config["residual"],
            "output_size": self.config["output_size"],
            "last_layer": self.config.get("last_layer", "linear"),
        }

    def _correct_checkpoint_keys(self, state_dict, model):
        """
        Adjusts the keys in a state_dict to align with the model’s expected keys.

        Args:
            state_dict (dict): Original checkpoint state_dict.
            model (torch.nn.Module): The model with expected keys.

        Returns:
            dict: Corrected state_dict.
        """
        corrected_state_dict = {}
        for key in state_dict:
            new_key = key.replace("linear", "last_layer").replace("tcn.network", "network").replace("tcn", "")
            corrected_state_dict[new_key] = state_dict[key]
        return corrected_state_dict

    def _correct_checkpoint_shape(self, model, state_dict):
        """
        Corrects the shape of checkpoint weights for compatibility with the current model.

        Args:
            model (torch.nn.Module): The model to load state into.
            state_dict (dict): The checkpoint with potential shape mismatches.
        """
        for key in state_dict.keys():
            if "last_layer.weight" in key:
                weight = state_dict[key]
                if isinstance(model.stage1.last_layer, torch.nn.Conv1d) and weight.dim() == 2:
                    # Add a channel dimension for Conv1d compatibility
                    state_dict[key] = weight.unsqueeze(2)
                elif isinstance(model.stage1.last_layer, torch.nn.Linear) and weight.dim() == 3:
                    # Remove the extra dimension for Linear compatibility
                    state_dict[key] = weight.squeeze(2)
        model.load_state_dict(state_dict)
