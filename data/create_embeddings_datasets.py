#!/usr/bin/env python

"""
The code saves in the output_folder a pkl file for every video in the dataset.
Every pkl file contains a dict with at "video_embeddings" a numpy array containing the embedded video by a feature
encoder, this can be of shape [1, temporal_size, latent_size] or [n_augmentations, temporal_size, latent_size];
and a list of frame image names at key "image_names".

Usage:  python3 create_embeddings_datasets.py -parFile data/emb_datatasets_public_1x.yml
"""
import os
import sys
import argparse
import yaml
import pickle
import pdb
import numpy as np
import pandas as pd

from src.utils import io


def main(args):
    # Read params from input yml file
    if not os.path.exists(args.par_file):
        raise Exception("Parameter file %s does not exist" % (args.par_file))
    print("Parsing parfile: %s " % (args.par_file))
    with open(args.par_file, 'r') as stream:
        pars = yaml.safe_load(stream)
    print("Input pars: \n", pars)

    for dataset_dict in pars:
        print("Working on dataset: ", dataset_dict)
        dataset_dict_pars = pars[dataset_dict]

        # Where to save output dataset
        dataset_dict_pars["output_folder"] = (dataset_dict_pars["output_folder"])
        os.makedirs(dataset_dict_pars["output_folder"], exist_ok=True)

        # Check that the same number of videos of the dataset list were processed by feature extraction
        if dataset_dict_pars["video_list"] == "":
            # pick them in in the csvs_path folder
            list_of_video_names = [f for f in os.listdir(dataset_dict_pars["video_csv_intermediate_path"])
                                   if os.path.isfile(os.path.join(dataset_dict_pars["video_csv_intermediate_path"], f))
                                   and ".csv" in f]
        else:
            with open((dataset_dict_pars["video_list"]), "r") as f:
                list_of_video_names = [line.rstrip() for line in f]
        for i_na in range(dataset_dict_pars["number_of_augmentations"]):
            current_folder_feature_extraction = os.path.join(
                (dataset_dict_pars["feature_extraction_folder"]),
                str(i_na))
            print("Retrieving data from: ", current_folder_feature_extraction)
            filenames = [file for file in
                         os.listdir(current_folder_feature_extraction) if "pkl" in file]
            if len(list_of_video_names) != len(filenames):
                set1 = set([f[:-4] for f in list_of_video_names])
                set2 = set([f[:-4] for f in filenames])
                missing = list(sorted(set1 - set2))
                print("Going in debug mode. List of video with missing augmentations: ", missing)
                pdb.set_trace()

        # Loop over the video filenames in the dataset and transform the pkl files saved from feature_extraction
        # to pkls in the right format for classification tcn app dataloader (EmbeddingsDataset). One video at a time.
        for fn in list_of_video_names:
            vn = fn[:-4]
            print("Working on video: ", vn)

            # Loop over the number of augmentations performed for each video
            list_of_video_embedding = []
            list_of_list_of_image_names = []
            for i_na in range(dataset_dict_pars["number_of_augmentations"]):
                current_folder_feature_extraction = os.path.join(
                    (dataset_dict_pars["feature_extraction_folder"]),
                    str(i_na))

                # Process pkl files containing the output of feature extraction
                temp_list_fn = []
                temp_list_embs = []
                file = os.path.join(current_folder_feature_extraction, vn + ".pkl")
                with open(file, 'rb') as fl:
                    while True:
                        try:
                            pickle_data = pickle.load(fl)
                            temp_list_fn.append(pickle_data[0])
                            temp_list_embs.append(pickle_data[1])
                        except:
                            break

                    # Get two lists: a list of unique frame paths and a list with their corresponding embeddings
                    used = set()
                    unique_temp_list_fn = [x for x in temp_list_fn if x not in used and (used.add(x) or True)]
                    used = set()
                    unique_temp_list_embs = [temp_list_embs[i] for i, x in enumerate(temp_list_fn) if
                                             x not in used and (used.add(x) or True)]

                    # Sort by frame timestamp, it's only needed for very few videos, let's do it for all anyways
                    ids = [int(idfn[idfn.rfind('_') + 1:-4]) for idfn in unique_temp_list_fn]
                    new_order = sorted(range(len(ids)), key=lambda k: ids[k])
                    unique_temp_list_fn = [unique_temp_list_fn[i] for i in new_order]
                    unique_temp_list_embs = [unique_temp_list_embs[i] for i in new_order]
                    ids = [int(idfn[idfn.rfind('_') + 1:-4]) for idfn in unique_temp_list_fn]

                    # Check that frames ids after operations above are still in increasing temporal order
                    for i, id in enumerate(ids[:-1]):
                        if ids[i + 1] <= id:
                            print("Ids are not in increasing order!")
                            exit()
                    print("Feature encoding: total number of frames is %d, max timestamp is %d" % (
                        len(unique_temp_list_fn), np.max(ids)))

                    # process dataset in csv (input of classification_rnn)
                    csv_path = os.path.join((dataset_dict_pars["video_csv_intermediate_path"]),
                                            vn + ".csv")
                    csv_file = pd.read_csv(csv_path)
                    gt_ids = [int(id[id.rfind('_') + 1:-4]) for id in csv_file["full_path"].to_list()]
                    gt_ids = gt_ids[::dataset_dict_pars["sampling_factor"]]
                    print("CSV total number of frames is %d, max timestamp is %d" % (len(gt_ids), np.max(gt_ids)))

                    # Delete frames that are not in the GT, this can happen if you did the feature extraction at a smaller
                    # temporal subsampling that dataset_dict_pars["sampling_factor"], ie you have more ids than gt_ids
                    unique_temp_list_embs = [unique_temp_list_embs[i] for i, id in enumerate(ids) if id in gt_ids]
                    unique_temp_list_fn = [unique_temp_list_fn[i] for i, id in enumerate(ids) if id in gt_ids]
                    ids = [id for id in ids if id in gt_ids]

                    # Check that number of encoded frames and last timestamp match between GT and feat extraction output
                    if abs(len(gt_ids) - len(unique_temp_list_fn)) > dataset_dict_pars["sampling_factor"] or \
                            abs(np.max(ids) - np.max(gt_ids)) > dataset_dict_pars["sampling_factor"] * (
                            gt_ids[1] - gt_ids[0]):
                        print("Number of encoded frames or max timestamp are not as expected. Going in debug mode!")
                        pdb.set_trace()

                    list_of_video_embedding.append(np.stack(unique_temp_list_embs))
                    list_of_list_of_image_names.append([os.path.basename(v) for v in unique_temp_list_fn])

            if dataset_dict_pars["number_of_augmentations"] > 1:
                if list_of_list_of_image_names[1] == list_of_list_of_image_names[0] == list_of_list_of_image_names[2] == \
                        list_of_list_of_image_names[3] == list_of_list_of_image_names[4]:
                    print("Sanity check for augmentations is OK. "
                          "Different augmentations have the same list of image frames.")
                else:
                    print("List of image frames are not identical for different augmentations. Going in debug mode!")
                    pdb.set_trace()

            # Save output pkl in the right format for classification tcn app dataloader (EmbeddingsDataset)
            output_for_video = {"video_embeddings": np.stack(list_of_video_embedding),
                                "image_names": list_of_list_of_image_names[0]}
            io.write_pickle(os.path.join(dataset_dict_pars["output_folder"], vn + ".pkl"), output_for_video)

        print("Dataset creation ended without issues.")

    print("All datasets done.")


if __name__ == '__main__':
    # Parse input and start main
    parser = argparse.ArgumentParser(description='Create an embeddings dataset')
    parser.add_argument('-parFile', action='store', dest='par_file',
                        help='path to the parameter file', required=True)
    args = parser.parse_args()
    main(args)
