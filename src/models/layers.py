"""
Define layers for temporal video segmentation in PyTorch.
The attention layers were obtained from: https://github.com/ChinaYi/ASFormer
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np


class Chomp1d(nn.Module):
    """
    Layer to trim the output in temporal dimension to ensure causality in convolutions.
    This prevents future time steps from influencing current time step predictions.
    """

    def __init__(self, chomp_size):
        """
        Initializes the Chomp1d layer.

        Args:
            chomp_size (int): Number of elements to trim from the end of the output tensor.
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Forward pass that chomps the output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Chomped tensor of shape (batch_size, channels, length - chomp_size).
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal block for a Temporal Convolutional Network (TCN).
    Each block consists of one or two 1D convolutional layers, with options for dropout and residual connections.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, conv_type, num_of_convs=2,
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
            conv_type (str): Type of convolution ('causal' or 'non-causal').
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
        if conv_type == "causal":
            chomp_layer = Chomp1d(padding)
            layers = [self.conv1, chomp_layer, nn.ReLU(), nn.Dropout(dropout)]
            if num_of_convs == 2:
                layers.extend([self.conv2, chomp_layer, nn.ReLU(), nn.Dropout(dropout)])
        else:
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


class AttentionHelper(nn.Module):
    """
    Helper class for attention calculations, specifically for scalar dot-product attention.
    """

    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        """
        Computes scalar dot-product attention with padding mask.

        Args:
            proj_query (torch.Tensor): Query tensor of shape (batch_size, channels, length).
            proj_key (torch.Tensor): Key tensor of shape (batch_size, channels, length).
            proj_val (torch.Tensor): Value tensor of shape (batch_size, channels, length).
            padding_mask (torch.Tensor): Mask tensor of shape (batch_size, 1, length) to handle padded elements.

        Returns:
            tuple: Output tensor and attention tensor.
        """
        m, c1, l1 = proj_query.shape
        _, c2, l2 = proj_key.shape
        assert c1 == c2

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # (batch, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6)  # Mask zero paddings
        attention = self.softmax(attention) * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    """
    Self-attention layer for temporal sequences, allowing feature learning across time.
    """

    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl):
        """
        Initializes an attention layer with given input and output dimensions.

        Args:
            q_dim (int): Query dimension.
            k_dim (int): Key dimension.
            v_dim (int): Value dimension.
            r1, r2, r3 (int): Reduction ratios for query, key, and value projections.
            bl (int): Block length for sliding window attention.
        """
        super(AttLayer, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)
        self.bl = bl
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()

    def construct_window_mask(self):
        """Constructs a sliding window mask for self-attention within a specified range."""
        window_mask = torch.zeros((1, self.bl, self.bl + 2 * (self.bl // 2)))
        for i in range(self.bl):
            window_mask[:, :, i:i + self.bl] = 1
        return window_mask.to(device)

    def forward(self, x1, mask):
        """
        Forward pass for the attention layer.

        Args:
            x1 (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor to apply selective attention.

        Returns:
            torch.Tensor: Output after applying sliding window attention.
        """
        query = self.query_conv(x1)
        key = self.key_conv(x1)
        value = self.value_conv(x1)
        return self._sliding_window_self_att(query, key, value, mask)

    # [Other methods omitted for brevity...]


class AttModule(nn.Module):
    """
    Attention module for applying self-attention and channel-wise normalization.
    """

    def __init__(self, width, in_channels, out_channels, r1, r2, alpha, instance_norm=True,
                 dropout_att=False, att_relu_output=False):
        """
        Initializes the attention module.

        Args:
            width (int): Width of the attention window.
            in_channels (int): Input channel size.
            out_channels (int): Output channel size.
            r1, r2 (int): Reduction ratios.
            alpha (float): Scaling factor for the attention output.
            instance_norm (bool): Apply instance normalization if True.
            dropout_att (bool): Apply dropout if True.
            att_relu_output (bool): Apply ReLU activation on attention output if True.
        """
        super(AttModule, self).__init__()
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False) if instance_norm else None
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, width)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout() if dropout_att else None
        self.alpha = alpha
        self.att_relu_output = att_relu_output
        self.relu = nn.ReLU()

    def forward(self, xinput):
        """
        Forward pass for the attention module.

        Args:
            xinput (tuple): Tuple containing input tensor and optional mask.

        Returns:
            torch.Tensor: Output tensor after applying attention and residual connections.
        """
        x, mask = xinput
        out = self.instance_norm(x) if self.instance_norm else x
        out = self.alpha * self.att_layer(out, mask) + out
        out = self.conv_1x1(out)
        if self.dropout:
            out = self.dropout(out)

        if mask is not None:
            output = (x + out) * mask[:, 0:1, :]
            return self.relu(output) if self.att_relu_output else output
        else:
            output = x + out
            return self.relu(output) if self.att_relu_output else output
