#!/usr/bin/env python

""" Define, build, and compile a frame encoding network using PyTorch.
"""

import sys
import torch
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class FrameEncoderModel:
    """
    Class to define, build, and exploit a PyTorch frame classification model.
    """

    def __init__(self, encoder_type="resnet50", verbosity_level=0):
        """
        Initialize the frame encoder model.

        Args:
            verbosity_level (int): Defines the amount of processing details (from 0 to 2).
        """
        self.encoder_type = encoder_type
        self.verbosity_level = verbosity_level
        self.model = None

    def create_model(self):
        """
        Instantiate the model according to the selected encoder.
        """
        if self.encoder_type == "resnet50":
            model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(model.children())[:-1])  # Output: [B, 2048, 1, 1]
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

    def resume_model(self, resume_path):
        """
        Resume torch model from a saved state.

        Args:
            resume_path (str): Path to the model to load.
        """
        self.model = torch.load(resume_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def run_single_frame(self, img_chw):
        """
        Run model on a single frame to retrieve its feature encoding.

        Args:
            img_chw (np.array): CHW-format image (float32, range [0, 1]).

        Returns:
            np.array: Feature representation of the frame.
        """
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.encoder_type == "resnet50":
            c, h, w = img_chw.shape
            if self.verbosity_level > 0:
                print(f" -- Running ResNet50 inference. Input shape: {c, h, w}")
            net_img = np.reshape(img_chw, (1, c, h, w))
            with torch.no_grad():
                x = torch.from_numpy(net_img.astype(np.float32)).to(device)
                outputs = self.model(x).squeeze().cpu().numpy()
            if self.verbosity_level > 1:
                print("Feature vector:", outputs)
            return outputs

        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
