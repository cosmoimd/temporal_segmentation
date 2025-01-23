#!/usr/bin/env python

""" Define, build, and compile a frame encoding network using PyTorch.
"""

import sys
import torch
import os
import numpy as np


class FrameEncoderModel:
    """
    Class to define, build, and exploit a PyTorch frame classification model.
    """

    def __init__(self, verbosity_level=0):
        """
        Initialize the frame encoder model.

        Args:
            verbosity_level (int): Defines the amount of processing details (from 0 to 2).
        """
        self.verbosity_level = verbosity_level
        self.model = None

    def create_model(self):
        """
        Instantiate the deep model that will perform frame classification using a ResNet50 architecture.
        """
        self.model = resnet50(pretrained=True)  # Adjust this to your specific function call if different

    def resume_model(self, resume_path):
        """
        Resume torch model from a saved state.

        Args:
            resume_path (str): Path to the model to load.
        """
        self.model = torch.load(resume_path)

    def run_single_frame(self, img_chw):
        """
        Run model on a single frame to retrieve its feature encoding.
        This function will be used by the device live, processing one frame at a time.

        Returns:
            np.array: Feature representation of the frame.
        """
        c, h, w = img_chw.shape
        if self.verbosity_level > 0:
            print(f" -- Running Frame Encoder PyTorch inference. Input shape was {c, h, w}")
        net_img = np.reshape(img_chw, (1, c, h, w))
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(net_img.astype(np.float32)).cuda()
            outputs = self.model(x)
            outputs = outputs.cpu().numpy()
        if self.verbosity_level > 1:
            print("Feature vector for input frame is:", outputs)

        return outputs
