#!/usr/bin/env python

"""
The code saves in the output_folder a pkl file for every video in the dataset containing its video frame features,

Usage:  CUDA_VISIBLE_DEVICES=0 python3 feature_extraction.py --config ymls/feature_extraction_1x_RC.yml
Usage:  CUDA_VISIBLE_DEVICES=4 python3 feature_extraction.py --config ymls/feature_extraction_5x_aug_RC.yml
"""

import os
import sys
import random
import argparse
import yaml
import torch
import shutil
import pickle
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Ensure we can import the FrameEncoderModel
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../../')
sys.path.insert(0, project_root)

from data.feature_extraction.frame_encoder_model import FrameEncoderModel

def get_user_defined_augmentation_transform():
    """Generates augmentation transforms for video frames."""
    return transforms.RandomChoice([
        transforms.ColorJitter(
            brightness=random.uniform(0.9, 1.1),
            contrast=random.uniform(0.9, 1.1),
            saturation=random.uniform(0.9, 1.1),
            hue=random.uniform(0.0, 0.1)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5)
    ])


class VideoFrameDataset(Dataset):
    """Dataset class for loading video frames from a directory using frame names from a CSV file."""

    def __init__(self, video_name, csv_files_dir, root_dir,
                 augment=False, subsample=5, encoder_type="resnet50"):
        """
        Args:
            video_name (str): Video identifier (matches CSV and frame folder).
            csv_files_dir (str): Directory containing CSV files.
            root_dir (str): Directory containing all the frames.
            augment (bool): Whether to apply data augmentation.
            subsample (int): Keep every 'subsample' frame (default: 5 for 5FPS).
        """
        self.frame_info = pd.read_csv(os.path.join(csv_files_dir, video_name + ".csv"), usecols=[0])
        self.encoder_type = encoder_type

        # Subsample: Keep every 5th frame (if subsample=5)
        self.frame_info = self.frame_info.iloc[::subsample].reset_index(drop=True)

        # Fix the bug with 001-012 video frame names
        if video_name == '001-012':
            self.frame_info.iloc[:, 0] = self.frame_info.iloc[:, 0].str.replace('.jpg', '.0.jpg', regex=False)

        self.root_dir = os.path.join(root_dir, video_name + "_frames")
        self.augment = augment

        # Remove missing images before DataLoader uses them
        missing_files = []
        valid_images = []

        for img_name in self.frame_info.iloc[:, 0]:
            full_path = os.path.join(self.root_dir, img_name)
            if os.path.exists(full_path):
                valid_images.append(img_name)
            else:
                missing_files.append(full_path)

        # Save only valid images
        self.frame_info = pd.DataFrame(valid_images, columns=["frame_filename"])

        base_transforms = [transforms.Resize((224, 224)), transforms.ToTensor()]

        # Only normalize if using ResNet50
        if self.encoder_type == "resnet50":
            base_transforms.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))

        if self.augment:
            base_transforms.insert(0, get_user_defined_augmentation_transform())

        self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.frame_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame_info.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        return image, img_name


def main(args):
    """Main function to run feature extraction."""

    # Load config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    output_folder = config["general"]["output_folder"]
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy2(args.config, output_folder)

    # Load dataset paths
    annotation_csv_path = config["data_loader"]["annotation_csv_path"]
    dataset_path = config["data_loader"]["dataset_path"]
    augmentation = config["data_loader"]["augmentation"]

    list_of_videos = [v[:-4] for v in os.listdir(annotation_csv_path) if v.endswith('.csv')]

    # Load the encoder model
    encoder_type = config["model"].get("encoder", "resnet50")
    encoder = FrameEncoderModel(encoder_type=encoder_type)
    encoder.create_model()
    encoder.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.model.to(device)

    for video_name in list_of_videos:
        print(f"Processing video: {video_name}")

        dataset = VideoFrameDataset(video_name,
                                    annotation_csv_path,
                                    dataset_path,
                                    subsample=config["data_loader"]["subsampling"],
                                    augment=augmentation,
                                    encoder_type=encoder_type)
        data_loader = DataLoader(dataset,
                                 batch_size=config["data_loader"]["batch_size"],
                                 shuffle=False,
                                 num_workers=config["data_loader"]["num_workers"])

        for loop_number in range(config["general"]["n_of_augmentations"]):
            print(f"Feature encoding loop number: {loop_number + 1}")

            # Initialize storage for all encoded frames and corresponding image names
            encoded_results = []

            for batch_data, img_names in tqdm(data_loader, desc=f"Extracting Features Loop {loop_number + 1}"):
                # Skip missing images
                if batch_data is None:
                    continue

                batch_data = batch_data.to(device)

                # Encode features
                for i in range(len(img_names)):
                    frame = batch_data[i].cpu().numpy()
                    feature_vector = encoder.run_single_frame(frame)
                    encoded_results.append((feature_vector, img_names[i]))

            # Save all features for this video + augmentation round
            save_path = os.path.join(output_folder, f"{video_name}_{loop_number}.pkl")
            with open(save_path, 'wb') as fpkl:
                pickle.dump(encoded_results, fpkl)

            print(
                f"Saved {len(encoded_results)} encoded frames for {video_name}, augmentation {loop_number} at {save_path}")

    print(f"All videos processed. Encoded features saved in {output_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature extraction for video dataset.")
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)
