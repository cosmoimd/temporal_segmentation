"""
Run inference with a temporal segmentation model using the ModelFactory.

Usage:
CUDA_VISIBLE_DEVICES=0 python src/inference.py -parFile ymls/inference/inference_colontcn_4fold.yml
"""
import argparse
import yaml
import torch
import sys
import os
import pickle
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

from src.models.factory import ModelFactory
from src.data_loader.embeddings_dataset import EmbeddingsDataset, custom_collate_batch
from src.testing import testing

def main(args):
    """
    Main function to run inference using the specified configuration.

    Args:
        args: Command-line arguments containing the path to the parameter file.
    """
    # Load configuration
    if not os.path.exists(args.parFile):
        raise FileNotFoundError(f"Parameter file {args.parFile} does not exist")

    print(f"Parsing configuration file: {args.parFile}")
    with open(args.parFile, 'r') as stream:
        pars = yaml.safe_load(stream)

    print("Starting inference with the following parameters:\n", pars)
    output_folder = pars["general"].get("output_folder", "output")
    os.makedirs(output_folder, exist_ok=True)

    # Load model using ModelFactory
    model_factory = ModelFactory(pars["model"])
    model = model_factory.create_model()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create inference dataset and loader
    inference_dataset = EmbeddingsDataset(
        datasets_params=pars["data_loader"].get("valid"),
        phase="inference",
        rc_csv=pars["data_loader"]["rc_csv"],
        prepare_dataset=pars["data_loader"].get("prepare_dataset", False),
        temp_folder=pars["data_loader"].get("temp_folder", None),
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=1,
        num_workers=1,
        collate_fn=custom_collate_batch,
        shuffle=False,
        pin_memory=False
    )

    # Run inference
    with torch.no_grad():
        for batch_idx, (matrix, _, video_name, list_of_frame_names, _) in enumerate(inference_loader):
            import pdb
            pdb.set_trace()
            net_input = matrix.to(device).transpose(1, 2)  # Adjust input dimensions
            model_output = model(net_input)[0] # Use the last stage output

            video_name = video_name[0]
            list_of_frame_names = list_of_frame_names[0]

            print(f"Running inference on video: {video_name}")

            # Save the output prediction for the current video
            output_path = os.path.join(output_folder, f"{video_name}.pkl")
            with open(output_path, "wb") as fp:
                pickle.dump({
                    "model_output": [model_output.detach().cpu().numpy()],
                    "list_of_frame_names": list_of_frame_names
                }, fp)

    print("Inference completed.")

    # run testing on inference output if a testing folder is specified
    if pars["general"]["testing_folder"] != "":
        pars["general"]["inference_output"] = pars["general"]["output_folder"]
        pars["general"]["output_folder"] = pars["general"]["testing_folder"]
        _ = testing(pars)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with a temporal segmentation model')
    parser.add_argument('-parFile', action='store', dest='parFile', required=True,
                        help='Path to the parameter file')
    args = parser.parse_args()
    main(args)
