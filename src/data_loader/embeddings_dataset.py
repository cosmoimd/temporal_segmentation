#!/usr/bin/env python

""" Utils for data loading and management of the video embeddings datasets.
"""

# Import built-in modules
from __future__ import print_function
import sys
import torch, torchvision
import torch.utils.data as data
import os
import numpy as np
import random
import pickle
import pandas as pd
import time
from collections import Counter

# Changes to the path
import utils

def custom_collate_batch(batch):
    """
    Custom batching to handle different temporally sized data (video).
    This is needed since different videos have different lengths.
    This function pads with zeros the input so as all the video embeddings have the same temporal size, which
    will be the same of the max temporal size in the batch. Predictions are padded with 999 values so that
    they can be ignored in loss computation. A binary mask for TCNs layers is also produced accordingly

    Args:
        batch (tuple): a tuple yielded at every iteration by an EmbeddingsDataset object

    Returns:
        x (torch.Tensor): (batch_size, temporal_size, feature_dimension) input embeddings matrix
        y (torch.Tensor): (batch_size, temporal_size): list of gts, one for each frame (temporal_size)
        batch_list_of_video_names (list): list of video names in the batch
        list_of_frames (list): list of frame names (temporal_size)
    """
    # retrieve max temporal size in the elements of the batch
    # this will define the output temporal size of the batch
    max_temporal_size = max([be[0].shape[0] for be in batch])

    # create padded batch by looping on the input batch
    batch_list_of_list_of_frames = []
    batch_list_of_video_names = []
    video_embeddings = torch.zeros((len(batch), max_temporal_size, batch[0][0].shape[1]))
    gts = torch.full((len(batch), max_temporal_size), 999)
    mask = torch.zeros((len(batch), max_temporal_size))
    for i, (emb, list_of_gts, video_name, list_of_frames) in enumerate(batch):
        # force embeddings to all have the same temporal size
        # in order to do that concatenate before and after additional frame videos
        # [size_1,video_lengths,size_2], pad each video with size_1 elements on the left, and size_2 on the right
        size_1 = int((max_temporal_size - emb.shape[0]) / 2)
        size_2 = max_temporal_size - (emb.shape[0] + size_1)

        # fill input video embeddings, gts and masks according to required padding
        video_embeddings[i, size_1:emb.shape[0] + size_1, :] = torch.from_numpy(emb)
        gts[i, size_1:emb.shape[0] + size_1] = torch.from_numpy(np.array(list_of_gts))

        mask[i, size_1:emb.shape[0] + size_1] = 1.

        # append list of frame names and video name so that they can be used at inference
        batch_list_of_list_of_frames.append([""] * size_1 + list_of_frames + [""] * size_2)
        batch_list_of_video_names.append(video_name)

    y = gts.long()
    mask = torch.Tensor(mask)
    return video_embeddings, y, batch_list_of_video_names, batch_list_of_list_of_frames, mask


def custom_collate_batch_training(batch):
    """
    Custom batching to handle different temporally sized data (video).
    This is needed since different videos have different lengths.
    This function pads with zeros the input so as all the video embeddings have the same temporal size, which
    will be the same of the max temporal size in the batch. Predictions are padded with 999 values so that
    they can be ignored in loss computation. A binary mask for TCNs layers is also produced accordingly

    Args:
        batch (tuple): a tuple yielded at every iteration by an EmbeddingsDataset object

    Returns:
        x (torch.Tensor): (batch_size, temporal_size, feature_dimension) input embeddings matrix
        y (torch.Tensor): (batch_size, temporal_size): list of gts, one for each frame (temporal_size)
        batch_list_of_video_names (list): list of video names in the batch
        list_of_frames (list): list of frame names (temporal_size)
    """
    # retrieve max temporal size in the elements of the batch
    # this will define the output temporal size of the batch
    max_temporal_size = max([be[0].shape[0] for be in batch])

    # create padded batch by looping on the input batch
    video_embeddings = torch.zeros((len(batch), max_temporal_size, batch[0][0].shape[1]))
    gts = torch.full((len(batch), max_temporal_size), 999)
    mask = torch.zeros((len(batch), max_temporal_size))
    for i, (emb, list_of_gts, video_name, list_of_frames) in enumerate(batch):
        # force embeddings to all have the same temporal size
        # in order to do that concatenate after additional frame videos
        # fill input video embeddings, gts and masks according to required padding
        video_embeddings[i, 0:emb.shape[0], :] = torch.from_numpy(emb)
        gts[i, 0:emb.shape[0]] = torch.from_numpy(np.array(list_of_gts))

        mask[i, 0:emb.shape[0]] = 1.

    # export torch tensors
    y = gts.long()
    mask = torch.Tensor(mask)

    return video_embeddings, y, None, None, mask


class EmbeddingsDataset(data.Dataset):
    """
    Construct a dataset of embeddings.
    Dataset v1:  Way to load datasets before 08/2022 - the feature extraction code has been changed after. It will be
     remove in the near future.
    Dataset v2:  Way to load datasets AFTER 08/2022 - the feature extraction code has been changed, dataset creation
     either, and data augmentation will be added.
    Each video in this dataset is represented by a 2D matrix of shape (time,latent_size).
    """
    def __init__(self, datasets_params, temp_folder, rc_csv, phase, prepare_dataset=True,
                 temporal_augmentation=False):
        """
            Init data loader.

        Args:
            datasets_params (dict): a dictionary containing datasets metadata and info for the dataloader
            temporal_augmentation (bool): whether to apply or not temporal augmentation to the input
        """
        self.len_of_videos = None
        self.temp_folder = temp_folder
        self.datasets_params = datasets_params
        self.file_names = None
        self.rc_csv = rc_csv
        self.temporal_augmentation = temporal_augmentation
        self.phase = phase
        self.fps = []
        self.video_names = []
        self.subsampling_factors = []
        self.load_mode = "numpy"

        # load datasets into data loader object
        self.prepare_dataset(prepare_dataset)

    def compute_weights(self):
        """
        Automatically compute class weights for CE and other loss function depending on the dataset loaded for training.
        """
        gts = []
        counter = 0

        # Loop over all datasets
        for dataset_number in range(len(self.datasets_params)):
            # Read video names of the dataset
            with open(self.datasets_params[dataset_number]["video_list"], "r") as f:
                print('Loading {} dataset.'.format(self.datasets_params[dataset_number]["video_list"]))
                video_names = [line.rstrip() for line in f]
                video_names = [l[:-4] if ".pkl" in l or ".csv" in l else l for l in video_names]

            # Read gts for each video
            for vn in video_names:
                vid_pkl_path = os.path.join(
                    self.datasets_params[dataset_number]["pkl_path"],
                    vn + ".pkl")
                with open(vid_pkl_path, 'rb') as f:
                    pickle_data = pickle.load(f)
                list_of_frames = pickle_data["image_names"]

                # get the gts for those frames and read them
                gt_names_list = self.datasets_params[dataset_number]["gt_name"]
                csv_path = os.path.join(self.datasets_params[dataset_number]["csv_path"], vn + ".csv")
                csv_file = pd.read_csv(csv_path)
                list_of_gts = csv_file[csv_file["frame_filename"].isin(list_of_frames)][self.datasets_params[dataset_number]["gt_name"]].to_list()
                gts += [self.datasets_params[dataset_number]["str_to_idx"][v] for v in list_of_gts]

                counter += 1

        cardinality = Counter(gts)
        del cardinality[999]
        sorted_cardinality = [cardinality[key] for key in sorted(cardinality.keys())]
        total_samples = sum(sorted_cardinality)
        class_weights = [total_samples / count for count in sorted_cardinality]

        return class_weights

    def prepare_dataset(self, prepare_dataset):
        """
        Load dataset metadata, create temp pkl to be used by this data loader object.

        Args:
            prepare_dataset(bool):

        Returns:
            A list of video names, a list of video embeddings pkl paths, a list of video csv paths (containing gts etc),
            a list of subsampling factors to be applied to be applied to the video embedding, list of gt names in each
            csv, list of str_to_idx gt
        """
        if prepare_dataset:
            os.makedirs(os.path.join(self.temp_folder), exist_ok=True)
            os.makedirs(os.path.join(self.temp_folder, self.phase), exist_ok=True)

        rc_dataset_csv = pd.read_csv(self.rc_csv, low_memory=False)

        self.len_of_videos = 0
        for dataset_number in range(len(self.datasets_params)):
            # read video names of the dataset
            with open(self.datasets_params[dataset_number]["video_list"], "r") as f:
                print('Loading {} dataset.'.format(self.datasets_params[dataset_number]["video_list"]))
                video_names = [line.rstrip() for line in f]
                video_names = [l[:-4] if ".pkl" in l or ".csv" in l else l for l in video_names]

            # read embeddings anf gt
            for vn in video_names:
                self.video_names.append(vn)

                self.subsampling_factors.append(self.datasets_params[dataset_number]["subsampling_factor"])

                # Read csv with gts and gts names (in their columns)
                dataset_csv = pd.read_csv(os.path.join(self.datasets_params[dataset_number]["csv_path"], vn + ".csv"),
                                          low_memory=False)
                fps = rc_dataset_csv.loc[rc_dataset_csv['unique_video_name'] == vn, 'fps'].values[0]
                self.fps.append(fps)

                if prepare_dataset:
                    print("Loading data information for video: ", vn)

                    # Get video embeddings and frame/image names
                    vid_pkl_path = os.path.join(
                        self.datasets_params[dataset_number]["pkl_path"], vn + ".pkl")
                    with open(vid_pkl_path, 'rb') as f:
                        pickle_data = pickle.load(f)
                    video_embeddings = pickle_data["video_embeddings"]
                    list_of_frames = pickle_data["image_names"]

                    # Fix dataset small bug
                    if vn == "001-012":
                        list_of_frames = [f.replace('.0.jpg', '.jpg') for f in list_of_frames]

                    # Get the gts for those frames and read them
                    list_of_gts = dataset_csv[dataset_csv["frame_filename"].isin(list_of_frames)][self.datasets_params[dataset_number]["gt_name"]].to_list()
                    list_of_gts = [self.datasets_params[dataset_number]["str_to_idx"][v] for v in list_of_gts]

                    # Keep only embeddings coming that come from a filename of which potentially we have the GT,
                    # ie they are in a row of the loaded csv (this is necessary because action csvs have less rows
                    # than location csv, because all insertion frames are missing)
                    list_of_gts_file_names = dataset_csv["frame_filename"].to_list()
                    list_of_embeddings_file_names = list_of_frames
                    emb_frames_to_keep = [i for i, filename in enumerate(list_of_embeddings_file_names)
                                          if filename in list_of_gts_file_names]
                    list_of_frames = [filename for i, filename in enumerate(list_of_embeddings_file_names)
                                      if filename in list_of_gts_file_names]

                    len_list_of_gts = len(list_of_gts)
                    if len(emb_frames_to_keep) != len_list_of_gts:
                        print("Issue, going in debug mode...")
                    if len(video_embeddings.shape) != 3:
                        video_embeddings = video_embeddings.squeeze(-1).squeeze(-1)
                    video_embeddings = video_embeddings[:, emb_frames_to_keep, :]

                    op = os.path.join(self.temp_folder, self.phase,
                                      str(vn) + "_video_embeddings_" +
                                      str(self.datasets_params[dataset_number]["subsampling_factor"]) + ".npy")
                    np.save(op, np.array(video_embeddings))
                    op = os.path.join(self.temp_folder, self.phase, str(vn) + "_list_of_gts_" +
                                      str(self.datasets_params[dataset_number]["subsampling_factor"]) + ".npy")
                    np.save(op, np.array(list_of_gts))
                    op = os.path.join(self.temp_folder, self.phase, str(vn) + "_list_frame_ids.pkl")
                    with open(op, "wb") as fp:
                        pickle.dump(list_of_frames, fp)

                self.len_of_videos += 1

    def __len__(self):
        return self.len_of_videos

    def __getitem__(self, index):
        """
        Retrieve one element from the dataset

        Temporal augmentation is applied if specified. This samples video embeddings frames with a frame-wise prob. of
        1/temporal_subsampling_factor.

        Args:
            index (int): sampled video index

        Returns:
            video embeddings batch: (temporal_size, feature_dimension)
            gts (list): list of gts, one for each frame (temporal_size)
            video_name (str): video name
            list_of_frames (list): list of frame names (temporal_size)
        """
        # get video embeddings and frame/image names
        video_name = self.video_names[index]
        subsampling_factor = self.subsampling_factors[index]
        if len(self.fps) > 0:
            fps = self.fps[index]
        else:
            fps = 0

        ip = os.path.join(self.temp_folder, self.phase, str(video_name) + "_video_embeddings_" +
                          str(subsampling_factor) + ".npy")
        video_embeddings = np.load(ip)

        ip = os.path.join(self.temp_folder, self.phase, str(video_name) + "_list_of_gts_" +
                          str(subsampling_factor) + ".npy")
        gts = np.load(ip)
        if self.phase == "inference":
            ip = os.path.join(self.temp_folder, self.phase, str(video_name) + "_list_frame_ids.pkl")
            with open(ip, 'rb') as f:
                list_of_frames = pickle.load(f)
        else:
            list_of_frames = []

        # randomly keep one video dimension if many feature extractions have been performed on the video
        # this is supposed to be happening at training on augmented data only
        if len(video_embeddings.shape) == 3:
            video_embeddings = video_embeddings[random.randint(0, video_embeddings.shape[0] - 1), :, :]

        if subsampling_factor == 5 and fps > 26:
            subsampling_factor = 6

        if subsampling_factor == 1 and fps > 26:
            if self.phase == "inference":
                random_start = 0
            else:
                random_start = random.randint(0, 5)
            total_frames = video_embeddings.shape[0]
            mask = np.ones(total_frames, dtype=bool)
            mask[random_start::6] = False
            video_embeddings = video_embeddings[mask]
            gts = gts[mask]

            # Use list comprehension with zip to filter the list based on the mask
            list_of_frames = [frame for frame, m in zip(list_of_frames, mask) if m]

        # apply subsampling of the input
        if subsampling_factor > 1:
            if self.temporal_augmentation:
                # Compute temporal augmentation encoded in a list of frames to keep
                if random.uniform(0, 1) < 0.4:
                    # randomly sample input frames with a probability=1/subsampling_factor
                    frame_indices = [f for f in range(video_embeddings.shape[0])
                                     if random.uniform(0, 1) <= (1 / subsampling_factor)]
                else:
                    # keep equal distance between sampled frames
                    frame_indices = list(range(random.randint(0, subsampling_factor - 1),
                                               video_embeddings.shape[0],
                                               subsampling_factor))
                # Apply filtering
                video_embeddings = video_embeddings[np.array(frame_indices), :]
                gts = gts[np.array(frame_indices)]

                if self.phase == "inference":
                    list_of_frames = [list_of_frames[f] for f in frame_indices]
            else:
                # Apply evenly temporally spaced subsampling factor
                video_embeddings = video_embeddings[::subsampling_factor, :]
                gts = gts[::subsampling_factor]

            if self.phase == "inference":
                list_of_frames = list_of_frames[::subsampling_factor]

        gts = [gts]

        return video_embeddings, gts, video_name, list_of_frames

    def __len__(self):
        return self.len_of_videos
