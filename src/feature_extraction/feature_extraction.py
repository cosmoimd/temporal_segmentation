#!/usr/bin/env python
"""
Feature Extraction Script.

This script encodes video frames into their latent representations using a predefined encoder model.
It supports augmentation and can handle multiple videos in batches.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 feature_extraction.py --config data/feature_extraction_1x_RC.yml
    CUDA_VISIBLE_DEVICES=0 python3 feature_extraction.py --config data/feature_extraction_5x_aug_RC.yml
"""

import os
import argparse
import yaml
import shutil
from tqdm import tqdm
import pickle
from torchvision import transforms

from video_loader import get_video_loader
from frame_classification_model import FrameClassificationModel


def get_user_defined_augmentation_transform():
    """Generates augmentation transforms for video frames.

    Returns:
        dict: A dictionary containing frame-wise augmentation transforms.
    """
    return {
        "ColorJitter": transforms.ColorJitter(
            brightness=random.uniform(0.9, 1.1),
            contrast=random.uniform(0.9, 1.1),
            saturation=random.uniform(0.9, 1.1),
            hue=random.uniform(-0.05, 0.05)),
        "RandomVerticalFlip": transforms.RandomVerticalFlip(p=0.5),
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=0.5)
    }


def load_data(csv_file, cfg_data_loader):
    """Creates and returns a DataLoader for processing video frames based on CSV files.

    Args:
        csv_file (str): Path to the CSV file listing video frames.
        cfg_data_loader (dict): Configuration for the DataLoader.

    Returns:
        DataLoader: Configured video data loader.
    """
    root_dir = cfg_data_loader["video_frames_path"]  # Add this key to your config
    batch_size = cfg_data_loader["batch_size"]
    num_workers = cfg_data_loader["num_workers"]
    return get_video_loader(csv_file, root_dir, batch_size, num_workers)


def main(args):
    """Main function to run feature extraction.

    Args:
        args (Namespace): Command line arguments parsed by argparse.
    """
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Parameter file {args.config} does not exist.")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    os.makedirs(config["general"]["output_folder"], exist_ok=True)
    shutil.copy2(args.config, config["general"]["output_folder"])

    model = FrameClassificationModel(config["model"])
    if "model_path" in config["model"]:
        model.load_model(config["model"]["model_path"])
    model.to_device()

    csv_file = config["data_loader"]["video_list"]  # Path to the CSV file
    inference_loader = load_data(csv_file, config["data_loader"])

    for loop_number, batch_data in enumerate(tqdm(inference_loader, desc="Extracting Features")):
        print(f"Feature encoding loop number: {loop_number + 1}")
        batch, ids, labels, video_names = batch_data
        encoded_frames = model.encode(batch)

        # Save folder configuration
        save_folder = os.path.join(config["general"]["output_folder"], str(loop_number))
        os.makedirs(save_folder, exist_ok=True)

        for i_f, vn in enumerate(video_names):
            file_path = os.path.join(save_folder, f"{vn[:-4]}.pkl")
            with open(file_path, 'ab') as fpkl:
                pickle.dump((ids[i_f], encoded_frames[i_f]), fpkl)

    print(f"Done. Encoded videos saved at: {config['general']['output_folder']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run feature extraction on a video dataset.")
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)
