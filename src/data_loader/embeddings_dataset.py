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
current_folder = os.path.dirname(os.path.realpath(__file__))
modules_folder = os.path.join(current_folder, "./../..")
sys.path.append(modules_folder)

import utils

def get_cardinality(numbers):
    return Counter(numbers)


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
    if len(batch[0][1]) == 1:
        gts = torch.full((len(batch), max_temporal_size), 999)
    else:
        gts1 = torch.full((len(batch), max_temporal_size), 999)
        gts2 = torch.full((len(batch), max_temporal_size), 999)
    mask = torch.zeros((len(batch), max_temporal_size))
    for i, (emb, list_of_gts, video_name, list_of_frames, _) in enumerate(batch):
        # force embeddings to all have the same temporal size
        # in order to do that concatenate before and after additional frame videos
        # [size_1,video_lengths,size_2], pad each video with size_1 elements on the left, and size_2 on the right
        size_1 = int((max_temporal_size - emb.shape[0]) / 2)
        size_2 = max_temporal_size - (emb.shape[0] + size_1)

        # fill input video embeddings, gts and masks according to required padding
        video_embeddings[i, size_1:emb.shape[0] + size_1, :] = torch.from_numpy(emb)
        if len(list_of_gts) == 1:
            gts[i, size_1:emb.shape[0] + size_1] = torch.from_numpy(np.array(list_of_gts))
        else:
            gts1[i, size_1:emb.shape[0] + size_1] = torch.from_numpy(np.array(list(list_of_gts[0])))
            gts2[i, size_1:emb.shape[0] + size_1] = torch.from_numpy(np.array(list(list_of_gts[1])))
        mask[i, size_1:emb.shape[0] + size_1] = 1.

        # append list of frame names and video name so that they can be used at inference
        batch_list_of_list_of_frames.append([""] * size_1 + list_of_frames + [""] * size_2)
        batch_list_of_video_names.append(video_name)

    # export torch tensors
    if len(batch[0][1]) == 1:
        y = gts.long()
    else:
        y = [gts1.long(), gts2.long()]
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
    if len(batch[0][1]) == 1:
        gts = torch.full((len(batch), max_temporal_size), 999)
        if batch[0][4] is not None:  # check whether per-frame weighting is activated or not
            weights1 = torch.full((len(batch), max_temporal_size), 1)
    else:
        gts1 = torch.full((len(batch), max_temporal_size), 999)
        gts2 = torch.full((len(batch), max_temporal_size), 999)
        if batch[0][4] is not None:  # check whether per-frame weighting is activated or not
            weights1 = torch.full((len(batch), max_temporal_size), 1)
            weights2 = torch.full((len(batch), max_temporal_size), 1)
    mask = torch.zeros((len(batch), max_temporal_size))
    for i, (emb, list_of_gts, video_name, list_of_frames, weights) in enumerate(batch):
        # force embeddings to all have the same temporal size
        # in order to do that concatenate after additional frame videos
        # fill input video embeddings, gts and masks according to required padding
        video_embeddings[i, 0:emb.shape[0], :] = torch.from_numpy(emb)
        if len(list_of_gts) == 1:
            gts[i, 0:emb.shape[0]] = torch.from_numpy(np.array(list_of_gts))
            if weights:
                weights1[i, 0:emb.shape[0]] = torch.from_numpy(np.array(weights))
        else:
            gts1[i, 0:emb.shape[0]] = torch.from_numpy(np.array(list(list_of_gts[0])))
            gts2[i, 0:emb.shape[0]] = torch.from_numpy(np.array(list(list_of_gts[1])))
            if weights:
                weights1[i, 0:emb.shape[0]] = torch.from_numpy(np.array(list(weights[0])))
                weights2[i, 0:emb.shape[0]] = torch.from_numpy(np.array(list(weights[1])))
        mask[i, 0:emb.shape[0]] = 1.

    # export torch tensors
    if len(batch[0][1]) == 1:
        y = gts.long()
        if weights:
            weights = weights1.long()
    else:
        y = [gts1.long(), gts2.long()]
        if weights:
            weights = [weights1.long(), weights2.long()]
    mask = torch.Tensor(mask)

    return video_embeddings, y, None, None, mask, weights


class EmbeddingsDataset(data.Dataset):
    """
    Construct a dataset of embeddings.
    Dataset v1:  Way to load datasets before 08/2022 - the feature extraction code has been changed after. It will be
     remove in the near future.
    Dataset v2:  Way to load datasets AFTER 08/2022 - the feature extraction code has been changed, dataset creation
     either, and data augmentation will be added.
    Each video in this dataset is represented by a 2D matrix of shape (time,latent_size).
    """
    def __init__(self, datasets_params, phase, temp_folder, prepare_dataset=True, compute_boundary_weights=False,
                 temporal_augmentation=False, n_of_outputs=1, gaussian_noise=None, gaussian_noise_perc=0.5):
        """
            Init data loader.

        Args:
            datasets_params (dict): a dictionary containing datasets metadata and info for the dataloader
            temporal_augmentation (bool): whether to apply or not temporal augmentation to the input
            n_of_outputs (int): number of expected network outputs
            gaussian_noise (bool): whether to apply or not gaussian noise on the embeddings
            gaussian_noise_perc (float): probability at a frame level of applying random noise
        """
        self.len_of_videos = None
        self.temp_folder = temp_folder
        self.datasets_params = datasets_params
        self.file_names = None
        self.temporal_augmentation = temporal_augmentation
        self.n_of_outputs = n_of_outputs
        self.phase = phase
        self.fps = []
        self.video_names = []
        self.subsampling_factors = []
        self.load_mode = "numpy"
        self.compute_boundary_weights = compute_boundary_weights

        # Gaussian noise augmentation
        if gaussian_noise:
            if isinstance(gaussian_noise, float):
                self.gaussian_noise = gaussian_noise
            else:
                self.gaussian_noise = io.load_pickle(gaussian_noise)
        else:
            self.gaussian_noise = gaussian_noise
        self.gaussian_noise_perc = gaussian_noise_perc

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
                    core.resolvePathMacros(self.datasets_params[dataset_number]["pkl_path"]),
                    vn + ".pkl")
                with open(vid_pkl_path, 'rb') as f:
                    pickle_data = pickle.load(f)
                list_of_frames = pickle_data["image_names"]

                # get the gts for those frames and read them
                gt_names_list = self.datasets_params[dataset_number]["gt_name"]
                if self.n_of_outputs == 1:
                    csv_path = os.path.join(self.datasets_params[dataset_number]["csv_path"][0], vn + ".csv")
                    csv_file = pd.read_csv(csv_path)
                    list_of_gts = csv_file[csv_file["image_name"].isin(list_of_frames)][gt_names_list[0]].to_list()
                    gts += [self.datasets_params[dataset_number]["str_to_idx"][0][v] for v in list_of_gts]
                elif self.n_of_outputs == 2:
                    csv_file_0 = pd.read_csv(
                        os.path.join(self.datasets_params[dataset_number]["csv_path"][0], vn + ".csv"))
                    csv_file_1 = pd.read_csv(
                        os.path.join(self.datasets_params[dataset_number]["csv_path"][1], vn + ".csv"))
                    list_of_gts = [
                        csv_file_0[csv_file_0["image_name"].isin(list_of_frames)][gt_names_list[0]].to_list(),
                        csv_file_1[csv_file_1["image_name"].isin(list_of_frames)][gt_names_list[1]].to_list()]
                    gts += [[
                        [self.datasets_params[dataset_number]["str_to_idx"][0][v] for v in list_of_gts[0]],
                        [self.datasets_params[dataset_number]["str_to_idx"][1][v] for v in list_of_gts[1]]]]

                counter += 1

        if self.n_of_outputs == 1:
            cardinality = get_cardinality(gts)
            del cardinality[999]
            sorted_cardinality = [cardinality[key] for key in sorted(cardinality.keys())]
            total_samples = sum(sorted_cardinality)
            class_weights = [total_samples / count for count in sorted_cardinality]
        else:
            # Handle double output case
            gts_0 = []
            gts_1 = []
            for el_gt in gts:
                gts_0 += el_gt[0]
                gts_1 += el_gt[1]

            cardinality = get_cardinality(gts_0)
            del cardinality[999]
            sorted_cardinality_0 = [cardinality[key] for key in sorted(cardinality.keys())]
            total_samples_0 = sum(sorted_cardinality_0)

            cardinality = get_cardinality(gts_1)
            del cardinality[999]
            sorted_cardinality_1 = [cardinality[key] for key in sorted(cardinality.keys())]
            total_samples_1 = sum(sorted_cardinality_1)

            total_samples = total_samples_0 + total_samples_1

            class_weights_0 = [total_samples / count for count in sorted_cardinality_0]
            class_weights_1 = [total_samples / count for count in sorted_cardinality_1]

            class_weights = [class_weights_0, class_weights_1]

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
            os.makedirs(os.path.join(self.temp_folder, self.phase), exist_ok=False)

        self.len_of_videos = 0
        for dataset_number in range(len(self.datasets_params)):
            self.datasets_params[dataset_number]["pkl_path"] = core.resolvePathMacros(
                self.datasets_params[dataset_number]["pkl_path"])
            self.datasets_params[dataset_number]["video_list"] = core.resolvePathMacros(
                self.datasets_params[dataset_number]["video_list"])
            self.datasets_params[dataset_number]["csv_path"] = [core.resolvePathMacros(csvp) for csvp in
                                                                self.datasets_params[dataset_number]["csv_path"]]

            # read video names of the dataset
            with open(self.datasets_params[dataset_number]["video_list"], "r") as f:
                print('Loading {} dataset.'.format(self.datasets_params[dataset_number]["video_list"]))
                video_names = [line.rstrip() for line in f]
                video_names = [l[:-4] if ".pkl" in l or ".csv" in l else l for l in video_names]

            # read embeddings anf gt
            for vn in video_names:
                self.video_names.append(vn)

                sf = self.datasets_params[dataset_number]["subsampling_factors"]
                self.subsampling_factors.append(sf)

                # Read csv with gts and gts names (in their columns)
                dataset_csvs = [pd.read_csv(os.path.join(csvp, vn + ".csv"), low_memory=False) for csvp in
                                self.datasets_params[dataset_number]["csv_path"]]
                if "fps" in dataset_csvs[0]:
                    self.fps.append(dataset_csvs[0]["fps"].iloc[0])
                if prepare_dataset:
                    print("Loading data information for video: ", vn)
                    gt_names_list = self.datasets_params[dataset_number]["gt_name"]

                    # Get video embeddings and frame/image names
                    vid_pkl_path = os.path.join(
                        core.resolvePathMacros(self.datasets_params[dataset_number]["pkl_path"]), vn + ".pkl")
                    with open(vid_pkl_path, 'rb') as f:
                        pickle_data = pickle.load(f)
                    video_embeddings = pickle_data["video_embeddings"]
                    list_of_frames = pickle_data["image_names"]

                    # Get the gts for those frames and read them
                    if self.n_of_outputs == 1:
                        list_of_gts = dataset_csvs[0][dataset_csvs[0]["image_name"].isin(list_of_frames)][
                            gt_names_list[0]].to_list()
                        list_of_gts = [self.datasets_params[dataset_number]["str_to_idx"][0][v] for v in list_of_gts]
                        list_of_weights = []

                        if self.compute_boundary_weights == "type1":
                            delta = 30
                            for i, gt in enumerate(list_of_gts):
                                low = max(0, i - delta)
                                high = min(len(list_of_gts) - 1, i + delta)
                                uno = all(ele == list_of_gts[i] for ele in list_of_gts[low:high])
                                list_of_weights.append(1 if uno else 0.3)

                    elif self.n_of_outputs == 2:
                        list_of_gts = [
                            dataset_csvs[0][dataset_csvs[0]["image_name"].isin(list_of_frames)][
                                gt_names_list[0]].to_list(),
                            dataset_csvs[1][dataset_csvs[1]["image_name"].isin(list_of_frames)][
                                gt_names_list[1]].to_list()]
                        list_of_gts = [
                            [self.datasets_params[dataset_number]["str_to_idx"][0][v] for v in list_of_gts[0]],
                            [self.datasets_params[dataset_number]["str_to_idx"][1][v] for v in list_of_gts[1]]]

                        if self.compute_boundary_weights is not False:
                            list_of_weights_0 = []
                            list_of_weights_1 = []

                            if self.compute_boundary_weights == "type1":
                                delta = 30
                                for i, gt in enumerate(list_of_gts[0]):
                                    low = max(0, i - delta)
                                    high = min(len(list_of_gts[0]) - 1, i + delta)
                                    uno = all(ele == list_of_gts[0][i] for ele in list_of_gts[0][low:high])
                                    due = all(ele == list_of_gts[1][i] for ele in list_of_gts[1][low:high])
                                    list_of_weights_0.append(1 if uno else 0.3)
                                    list_of_weights_1.append(1 if due else 0.3)
                                list_of_weights = [list_of_weights_0, list_of_weights_1]

                    else:
                        raise Exception("Wrong number of GTs for a dataset.")

                    # Keep only embeddings coming that come from a filename of which potentially we have the GT,
                    # ie they are in a row of the loaded csv (this is necessary because action csvs have less rows
                    # than location csv, because all insertion frames are missing)
                    list_of_gts_file_names = dataset_csvs[0]["image_name"].to_list()
                    list_of_embeddings_file_names = list_of_frames
                    emb_frames_to_keep = [i for i, filename in enumerate(list_of_embeddings_file_names)
                                          if filename in list_of_gts_file_names]
                    list_of_frames = [filename for i, filename in enumerate(list_of_embeddings_file_names)
                                      if filename in list_of_gts_file_names]

                    len_list_of_gts = len(list_of_gts[0]) if self.n_of_outputs == 2 else len(list_of_gts)
                    if len(emb_frames_to_keep) != len_list_of_gts:
                        print("Issue, going in debug mode...")
                        import pdb
                        pdb.set_trace()
                    if len(video_embeddings.shape) == 3:
                        video_embeddings = video_embeddings[:, np.array(emb_frames_to_keep), :]
                    elif len(video_embeddings.shape) == 2:
                        video_embeddings = video_embeddings[np.array(emb_frames_to_keep), :]
                    else:
                        print("Issue, going in debug mode...")
                        import pdb
                        pdb.set_trace()

                    start = time.time()
                    op = os.path.join(self.temp_folder, self.phase,
                                      str(vn) + "_video_embeddings_" +
                                      str(sf[0]) + ".npy")
                    np.save(op, np.array(video_embeddings))
                    op = os.path.join(self.temp_folder, self.phase, str(vn) + "_list_of_gts_" +
                                      str(sf[0]) + ".npy")
                    np.save(op, np.array(list_of_gts))
                    if self.compute_boundary_weights is not False:
                        op = os.path.join(self.temp_folder, self.phase, str(vn) + "_list_of_weights_" +
                                          str(sf[0]) + ".npy")
                        np.save(op, np.array(list_of_weights))
                    op = os.path.join(self.temp_folder, self.phase, str(vn) + "_list_frame_ids.pkl")
                    with open(op, "wb") as fp:
                        pickle.dump(list_of_frames, fp)
                    end = time.time()
                    print("Saving data in: ", end - start)
                    print("Saved video metadata.")

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
                          str(subsampling_factor[0]) + ".npy")
        video_embeddings = np.load(ip)
        ip = os.path.join(self.temp_folder, self.phase, str(video_name) + "_list_of_gts_" +
                          str(subsampling_factor[0]) + ".npy")
        gts = np.load(ip)
        weights = None
        if self.compute_boundary_weights is not False:
            ip = os.path.join(self.temp_folder, self.phase, str(video_name) + "_list_of_weights.npy")
            weights = np.load(ip)
        if self.phase == "inference" or self.phase == "validation":
            ip = os.path.join(self.temp_folder, self.phase, str(video_name) + "_list_frame_ids.pkl")
            with open(ip, 'rb') as f:
                list_of_frames = pickle.load(f)
        else:
            list_of_frames = []

        # randomly keep one video dimension if many feature extractions have been performed on the video
        # this is supposed to be happening at training on augmented data only
        if len(video_embeddings.shape) == 3:
            video_embeddings = video_embeddings[random.randint(0, video_embeddings.shape[0] - 1), :, :]

        if len(subsampling_factor) > 1:
            subsampling_factor = subsampling_factor[random.randint(0, len(subsampling_factor) - 1)]
        else:
            subsampling_factor = subsampling_factor[0]

        if subsampling_factor == 5 and fps > 26:
            subsampling_factor = 6

        if subsampling_factor == 1 and fps > 26:
            if self.phase == "inference" or self.phase == "validation":
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
                if self.n_of_outputs == 2:
                    gts = gts[:, np.array(frame_indices)]
                    if self.compute_boundary_weights is not False:
                        weights = weights[:, np.array(frame_indices)]
                else:
                    gts = gts[np.array(frame_indices)]
                    if self.compute_boundary_weights is not False:
                        weights = weights[np.array(frame_indices)]

                if self.phase == "inference":
                    list_of_frames = [list_of_frames[f] for f in frame_indices]
            else:
                # Apply evenly temporally spaced subsampling factor
                video_embeddings = video_embeddings[::subsampling_factor, :]
                if self.n_of_outputs == 2:
                    gts = gts[:, ::subsampling_factor]
                    if self.compute_boundary_weights is not False:
                        weights = weights[:, ::subsampling_factor]
                else:
                    gts = gts[::subsampling_factor]
                    if self.compute_boundary_weights is not False:
                        weights = weights[::subsampling_factor]

            if self.phase == "inference" or self.phase == "validation":
                list_of_frames = list_of_frames[::subsampling_factor]

        # Apply gaussian noise on the input embeddings
        if self.gaussian_noise and self.phase == "training":
            if isinstance(self.gaussian_noise, float):
                if random.uniform(0, 1.0) < 0.5:
                    # Create a random mask with 50% ones
                    mask = (np.random.rand(*video_embeddings.shape) < self.gaussian_noise_perc).astype(float)

                    # Generate Gaussian noise
                    noise = mask * np.random.normal(loc=0.0, scale=self.gaussian_noise, size=video_embeddings.shape)

                    # Add the masked noise to the video embeddings
                    video_embeddings += noise

        if self.n_of_outputs == 2:
            gts = [gts[0:1, :], gts[1:2, :]]
            if self.compute_boundary_weights is not False:
                weights = [weights[0:1, :], weights[1:2, :]]
        else:
            gts = [gts]
            if self.compute_boundary_weights is not False:
                weights = [weights]

        return video_embeddings, gts, video_name, list_of_frames, weights

    def __len__(self):
        return self.len_of_videos
