"""
Define layers for temporal video segmentation in PyTorch.
The attention layers were obtained from: https://github.com/ChinaYi/ASFormer
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np


class TemporalBlock(nn.Module):
    """
    Temporal block for a Temporal Convolutional Network (TCN).
    Each block consists of one or two 1D convolutional layers, with options for dropout and residual connections.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, num_of_convs=2,
                 dropout=0.2, residual=True):
        """
        Initializes a TemporalBlock.

        Args:
            n_inputs (int): Number of input channels.
            n_outputs (int): Number of output channels.
            kernel_size (int): Kernel size for the convolutional layers.
            stride (int): Stride for the convolutional layers.
            dilation (int): Dilation factor for the convolutional layers.
            padding (int): Padding size to maintain input-output dimensions.
            num_of_convs (int): Number of convolutions in the block (1 or 2).
            dropout (float): Dropout rate for regularization.
            residual (bool): Whether to use residual connections.
        """
        super(TemporalBlock, self).__init__()
        self.num_of_convs = num_of_convs
        self.residual = residual

        # Define convolutional layers with weight normalization
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        if num_of_convs == 2:
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))

        # Define network sequence based on convolution type
        layers = [self.conv1, nn.ReLU(), nn.Dropout(dropout)]
        if num_of_convs == 2:
            layers.extend([self.conv2, nn.ReLU(), nn.Dropout(dropout)])

        self.net = nn.Sequential(*layers)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        """Initializes weights with a normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        if self.num_of_convs == 2:
            self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass of the temporal block with an optional residual connection.

        Args:
            x (list): List containing:
                x[0] (torch.Tensor): Input tensor (batch_size, channels, length).
                x[1] (torch.Tensor, optional): Mask tensor (batch_size, length) to selectively apply TCN.

        Returns:
            list: Output list containing:
                [0] (torch.Tensor): Output tensor (batch_size, channels, length).
                [1] (torch.Tensor, optional): Mask tensor, if provided in input.
        """
        out = self.net(x[0])
        if self.residual:
            res = x[0] if self.downsample is None else self.downsample(x[0])
            out = self.relu(out + res)

        if len(x) == 1:
            return [out]
        else:
            return [out * x[1].unsqueeze(1), x[1]]
