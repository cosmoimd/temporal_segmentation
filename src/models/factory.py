# src/models/factory.py
import torch
from src.models.mstcn import MS_TCN
from src.models.asformer import ASFormer

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

    def create_model(self):
        """
        Instantiates a model based on the configuration.

        Returns:
            nn.Module: An instance of the specified model.

        Raises:
            ValueError: If the model type specified in the configuration is unsupported.
        """
        model_type = self.config.get("model_type")
        model_class = self._get_model_class(model_type)

        # Instantiate model with required configuration parameters
        model_params = self._extract_model_params(model_type)
        model = model_class(**model_params)

        # Load model state if a model path is provided
        if "model_path" in self.config:
            model = self._load_model_state(model, self.config["model_path"])

        model.to(self.device)
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
        model_classes = {
            "tcn": MS_TCN,
            "mstcn": MS_TCN,
            "colontcn": MS_TCN,
            "ms-colontcn": MS_TCN,
            "asformer": ASFormer
        }

        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_classes[model_type]

    def _extract_model_params(self, model_type):
        """
        Extracts model-specific parameters from the configuration.

        Args:
            model_type (str): The type of model for which to extract parameters.

        Returns:
            dict: Dictionary of parameters specific to the model.
        """
        common_params = {
            "input_size": self.config["input_size"],
            "list_of_features_sizes": self.config["list_of_features_sizes"],
            "conv_type": self.config["conv_type"],
            "num_of_convs": self.config["num_of_convs"],
            "kernel_size": self.config["kernel_size"],
            "dropout": self.config["dropout"],
            "residual": self.config["residual"],
            "output_size": self.config["output_size"]
        }

        # Add model-specific parameters based on model type
        if model_type in {"mstcn", "ms-colontcn"}:
            specific_params = {
                "mstcn_num_stages": self.config["mstcn_num_stages"],
                "mstcn_input_size": self.config["mstcn_input_size"],
                "mstcn_list_of_features_sizes": self.config["mstcn_list_of_features_sizes"],
                "mstcn_kernel_size": self.config["mstcn_kernel_size"],
                "mstcn_dropout": self.config["mstcn_dropout"],
                "mstcn_num_of_convs": self.config["mstcn_num_of_convs"],
                "mstcn_residual": self.config["mstcn_residual"],
            }
        elif model_type == "asformer":
            specific_params = {
                "num_decoders": self.config["num_decoders"],
                "num_layers": self.config["num_layers"],
                "r1": self.config["r1"],
                "r2": self.config["r2"],
                "num_f_maps": self.config["num_f_maps"],
                "num_classes": self.config["num_classes"],
                "channel_masking_rate": self.config.get("channel_masking_rate", 0),
            }
        else:
            specific_params = {}

        # Combine common and specific parameters
        return {**common_params, **specific_params}


    def _load_model_state(self, model, model_path):
        """
        Loads the model's state dictionary from a specified path, handling legacy checkpoints as needed.

        Args:
            model (torch.nn.Module): Model to load state into.
            model_path (str): Path to the model's state dictionary.

        Returns:
            torch.nn.Module: Model with loaded state.
        """
        model_checkpoint_path = model_path
        print("Reloading checkpoint from", model_checkpoint_path)

        try:
            checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found: {model_checkpoint_path}")

        # Adjust keys in the checkpoint to match the model's expected keys
        corrected_state_dict = self._correct_checkpoint_keys(checkpoint['model_state_dict'], model)

        try:
            model.load_state_dict(corrected_state_dict)
        except RuntimeError:
            self._correct_checkpoint_shape(model, corrected_state_dict)

        return model

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
            new_key = key

            # Rename "linear" layers to "last_layer" if they exist
            if "linear" in key:
                new_key = key.replace("linear", "last_layer")

            # Fix temporal block layer naming discrepancy: "net.4" in checkpoint -> "net.3" in model
            if ".net.4" in key:
                new_key = key.replace(".net.4", ".net.3")

            # Adjust "tcn.network" to "network" if present
            new_key = new_key.replace("tcn.network", "network").replace("tcn", "")

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
            if "last_layer" in key and "bias" not in key:
                weight = state_dict[key]
                state_dict[key] = weight.squeeze(1) if weight.dim() == 3 else weight.unsqueeze(2)
        model.load_state_dict(state_dict)
