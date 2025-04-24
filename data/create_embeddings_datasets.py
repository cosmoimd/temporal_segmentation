#!/usr/bin/env python

"""
The code saves in the output_folder a pkl file for every video in the dataset.
Every pkl file contains a dict with at "video_embeddings" a numpy array containing the embedded video by a feature
encoder, this can be of shape [1, temporal_size, latent_size] or [n_augmentations, temporal_size, latent_size];
and a list of frame image names at key "image_names".

Usage:  python3 create_embeddings_datasets.py -parFile ymls/emb_datasets_public_1x.yml
Usage:  python3 create_embeddings_datasets.py -parFile ymls/emb_datasets_public_5x.yml
Usage:  python3 create_embeddings_datasets.py -parFile ymls/emb_datasets_public_5x_aug.yml
"""
import os
import sys
import argparse
import yaml
import pickle
import pdb
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../')
sys.path.insert(0, project_root)

from src.utils import io


def get_pkl_files(folder):
    """Reads all .pkl files in a directory."""
    return [f for f in os.listdir(folder) if f.endswith('.pkl')]


def main(args):
    # Read params from input yml file
    if not os.path.exists(args.parFile):
        raise Exception(f"Parameter file {args.parFile} does not exist")
    print(f"Parsing parfile: {args.parFile}")

    with open(args.parFile, 'r') as stream:
        pars = yaml.safe_load(stream)
    print("Input pars: \n", pars)

    # Where to save output dataset
    os.makedirs(pars["output_folder"], exist_ok=True)

    # Get all .pkl files from the feature extraction folder
    list_of_video_names = get_pkl_files(pars["feature_extraction_folder"])

    # Loop over the video filenames in the dataset
    for fn in list_of_video_names:
        vn = fn[:7]  # Remove .pkl extension
        print(f"Working on video: {vn}")

        # Loop over the number of augmentations performed for each video
        list_of_video_embedding = []
        list_of_list_of_image_names = []

        for i_na in range(pars["number_of_augmentations"]):
            # Process pkl files containing the output of feature extraction
            temp_list_fn = []
            temp_list_embs = []

            file = os.path.join(pars["feature_extraction_folder"], vn + "_" + str(i_na) + ".pkl")

            if not os.path.exists(file):
                print(f"Warning: Missing feature extraction file {file}")
                continue  # Skip missing .pkl files

            try:
                with open(file, 'rb') as fl:
                    while True:
                        try:
                            pickle_data = pickle.load(fl)

                            # Ensure `pickle_data` is a list of (embedding, image_name) tuples
                            if isinstance(pickle_data, list):
                                for data_tuple in pickle_data:
                                    if len(data_tuple) == 2:
                                        temp_list_fn.append(data_tuple[1])  # Image filename
                                        temp_list_embs.append(data_tuple[0])  # Feature embedding
                                    else:
                                        print(f"Warning: Skipping malformed entry in {file}: {data_tuple}")
                            else:
                                print(f"Unexpected data format in {file}, skipping.")

                        except EOFError:
                            break  # Stop reading when end of file is reached

            except Exception as e:
                print(f"Error reading {file}: {e}")

            # Get unique frame names and their corresponding embeddings
            used = set()
            unique_temp_list_fn = [x for x in temp_list_fn if x not in used and (used.add(x) or True)]
            used = set()
            unique_temp_list_embs = [temp_list_embs[i] for i, x in enumerate(temp_list_fn) if
                                     x not in used and (used.add(x) or True)]

            # Sort by frame timestamp
            ids = [int(float(idfn[idfn.rfind('_') + 1:-4])) for idfn in unique_temp_list_fn]
            new_order = sorted(range(len(ids)), key=lambda k: ids[k])
            unique_temp_list_fn = [unique_temp_list_fn[i] for i in new_order]
            unique_temp_list_embs = [unique_temp_list_embs[i] for i in new_order]
            ids = [int(float(idfn[idfn.rfind('_') + 1:-4])) for idfn in unique_temp_list_fn]

            # Ensure frames are in increasing temporal order
            if any(ids[i] >= ids[i + 1] for i in range(len(ids) - 1)):
                print("Error: Frame IDs are not in increasing order!")
                exit()

            print(f"Feature encoding: total frames: {len(unique_temp_list_fn)}, max timestamp: {np.max(ids)}")

            # Load the dataset CSV
            csv_path = os.path.join(pars["video_csv_intermediate_path"], vn + ".csv")
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file missing for {vn}, skipping video.")
                continue  # Skip missing CSVs

            csv_file = pd.read_csv(csv_path)
            gt_ids = [int(id[id.rfind('_') + 1:-4]) for id in csv_file["frame_filename"].to_list()]
            gt_ids = gt_ids[::pars["sampling_factor"]]
            print(f"CSV total frames: {len(gt_ids)}, max timestamp: {np.max(gt_ids)}")

            # Keep only frames that exist in GT
            unique_temp_list_embs = [unique_temp_list_embs[i] for i, id in enumerate(ids) if id in gt_ids]
            unique_temp_list_fn = [unique_temp_list_fn[i] for i, id in enumerate(ids) if id in gt_ids]
            ids = [id for id in ids if id in gt_ids]

            # Ensure frame count and timestamps match between GT and feature extraction
            if abs(len(gt_ids) - len(unique_temp_list_fn)) > pars["sampling_factor"] or \
                    abs(np.max(ids) - np.max(gt_ids)) > pars["sampling_factor"] * (gt_ids[1] - gt_ids[0]):
                print("Mismatch in encoded frames or max timestamp. Debugging...")
                pdb.set_trace()

            list_of_video_embedding.append(np.stack(unique_temp_list_embs))
            list_of_list_of_image_names.append([os.path.basename(v) for v in unique_temp_list_fn])

        # Validate augmentations
        if pars["number_of_augmentations"] > 1:
            if all(list_of_list_of_image_names[i] == list_of_list_of_image_names[0] for i in range(len(list_of_list_of_image_names))):
                print("Augmentation check passed: Identical frame lists across augmentations.")
            else:
                print("Augmentation mismatch! Debugging...")
                pdb.set_trace()

        # Save the final pkl for classification
        output_for_video = {
            "video_embeddings": np.stack(list_of_video_embedding),
            "image_names": list_of_list_of_image_names[0]
        }
        output_pkl_path = os.path.join(pars["output_folder"], vn + ".pkl")
        io.write_pickle(output_pkl_path, output_for_video)
        print(f"Saved embeddings dataset for {vn} at {output_pkl_path}")

    print("Dataset creation completed successfully.")


if __name__ == '__main__':
    # Parse input and start main
    parser = argparse.ArgumentParser(description='Create an embeddings dataset')
    parser.add_argument('-parFile', action='store', dest='parFile', help='path to the parameter file', required=True)
    args = parser.parse_args()
    main(args)
