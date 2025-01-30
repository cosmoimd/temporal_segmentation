"""
Profiling Script for models defined in this repository for video segmentation tasks.
Models defined in a YAML configuration file and the code computes the following:
- The total number of parameters in the model.
- The receptive field size of the model in terms of timesteps.
- The Billion Floating Point Operations (BFLOPs) required for a forward pass.

### Example Usage:
    CUDA_VISIBLE_DEVICES=0 python src/profiling.py --config ymls/profiling/colontcn_4fold.yml
    CUDA_VISIBLE_DEVICES=0 python src/profiling.py --config ymls/profiling/colontcn_5fold.yml

### Arguments:
    --config: (str) Path to the YAML configuration file specifying model and profiling parameters.
"""

import sys
import os
import argparse
from fvcore.nn import FlopCountAnalysis
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

from src.models.factory import ModelFactory
from src.utils.io import load_nested_yaml, setup_logger


def main(config_path):
    """
    Main function for profiling.

    Args:
        config_path (str): Path to the profiling YAML configuration file.
    """
    pars = load_nested_yaml(config_path)

    # Initialize Logger
    logger = setup_logger('profiling_logger')
    logger.info("Starting Profiling")

    # Initialize ModelFactory and create the model
    model_factory = ModelFactory(pars["model"])
    model = model_factory.create_model()

    logger.info("Model initialized for profiling")

    # Profile the model, including computing receptive field
    profile_model(model, logger, pars["model"])


def compute_receptive_field(kernel_size, num_blocks, num_convs_per_block=2):
    """
    Computes the receptive field of a TCN model based on kernel size, dilation, and number of layers.

    Args:
        kernel_size (int): Kernel size used in each convolutional layer.
        num_blocks (int): Number of temporal blocks in the TCN.
        num_convs_per_block (int): Number of convolutions per block.

    Returns:
        int: The total receptive field in terms of the number of timesteps.
    """
    receptive_field = 1  # Start with the current time step

    # Loop over each block
    for i in range(num_blocks):
        dilation = 2 ** i  # Dilation doubles per block
        # Loop over each convolution within the block
        for _ in range(num_convs_per_block):
            # Increment receptive field
            receptive_field += (kernel_size - 1) * dilation
            # Dilation remains the same within the block

    return receptive_field


def profile_model(model, logger, config, input_size=(1, 2048, 100)):
    """
    Profiles the model by computing the number of parameters, BFLOPs, and receptive field.

    Args:
        model (torch.nn.Module): The model to be profiled.
        logger: Logger instance for logging information.
        config (dict): Model configuration from YAML file.
        input_size (tuple): The input tensor shape as (batch_size, input_channels, temporal_size).
    """
    # Log the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params:,}")

    # Calculate receptive field using model configuration parameters
    kernel_size = config["kernel_size"]
    num_blocks = len(config["list_of_features_sizes"])
    num_of_convs = config.get("num_of_convs", 2)
    receptive_field = compute_receptive_field(kernel_size, num_blocks, num_of_convs)
    logger.info(f"Model Receptive Field: {receptive_field} timesteps")

    # Move model to evaluation mode and set up a dummy input
    model.eval()
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)

    # Calculate FLOPs using FlopCountAnalysis
    flops = FlopCountAnalysis(model, dummy_input)
    bflops = flops.total() / 1e9  # Convert FLOPs to BFLOPs
    logger.info(f"Model BFLOPs: {bflops:.3f} GFLOPs")

    # Log profiling completed
    logger.info("Profiling completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run profiling on model with specified configuration.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file for profiling.")
    args = parser.parse_args()

    main(args.config)
