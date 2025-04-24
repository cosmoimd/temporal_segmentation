"""
Define ColonTCN architectures.
"""
import torch.nn as nn
from src.models import layers as model_layers


class TCN(nn.Module):
    """
    Temporal convolutional network (TCN) and ColonTCN video temporal segmentation.
    """

    def __init__(self, input_size, output_size=None, num_channels=None, num_of_convs=2,
                 kernel_size=7, dropout=0.2, residual=True, last_layer="linear"):
        """
        Initializes a TemporalConvNet model.

        Args:
            input_size (int): Number of input channels.
            output_size (int or None): Number of output classes. If None, no output layer is applied.
            num_channels (list of int): List of output channels for each temporal block.
            num_of_convs (int): Number of convolutions per temporal block.
            dropout (float): Dropout rate after each convolutional layer.
            residual (bool): Use residual connections within each temporal block.
            last_layer (str): Type of last layer, "linear" or "conv".
        """
        super(TCN, self).__init__()
        self.last_layer_type = last_layer
        self.output_size = output_size

        # Build temporal blocks, iteratively
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            padding = int((kernel_size - 1) * dilation_size / 2)

            layers.append(model_layers.TemporalBlock(
                in_channels, out_channels, kernel_size,
                num_of_convs=num_of_convs, stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout, residual=residual
            ))

        self.network = nn.Sequential(*layers)
        num_channels_final = num_channels[-1]


        # Define final output layer
        self.last_layer = (nn.Linear(num_channels_final, output_size) if self.last_layer_type == "linear"
                           else nn.Conv1d(num_channels_final, output_size, 1))


    def forward(self, x, mask=None):
        """
        Forward pass through TemporalConvNet.

        Args:
            x (Tensor): Input tensor of shape (batch_size, temporal_size, input_channels).
            mask (Tensor, optional): Mask tensor of shape (batch_size, temporal_size, input_channels) for selective application.

        Returns:
            Tensor: Output tensor of shape (batch_size, temporal_size, output_channels).
        """
        # Pass through the main TCN blocks
        output = self.network([x, mask])[0] * mask.unsqueeze(1) if mask is not None else self.network([x])[0]

        # No multiscale layer
        if self.last_layer_type == "linear":
            output = self.last_layer(output.permute(0, 2, 1)).float()
        else:
            output = self.last_layer(output).float().permute(0, 2, 1)

        # Apply final output layer if output_size is defined
        if self.output_size is not None:

            # Apply mask and sigmoid activation if specified
            if mask is not None:
                return output * mask.unsqueeze(2)
            else:
                return output
        else:
            # If no output layer, return TCN feature outputs directly
            return output

class ColonTCN(nn.Module):
    """
    ColonTCN model definition
    """

    def __init__(self, input_size, output_size, list_of_features_sizes, kernel_size, num_of_convs=2,
                 dropout=1.0, residual=True,  last_layer="linear"):
        """
        Initializes a ColonTCN model.

        Args:
            input_size (int): Number of input channels.
            output_size (int or list of int): Number of output channels or classes.
            list_of_features_sizes (list of int): Number of channels in the base TCN layers.
            kernel_size (int): Kernel size for convolutions.
            num_of_convs (int): Number of convolutions per block in base TCN.
            dropout (float): Dropout rate for the TCN layers.
            residual (bool): Whether to use residual connections within base TCN blocks.
            last_layer (str): Type of final layer ("linear" or "conv").
        """
        super(ColonTCN, self).__init__()

        # Apply 1D conv + non-linearity on the input
        self.conv_first_layer = nn.Conv1d(input_size, list_of_features_sizes[0], 1)
        self.conv_first_layer_relu = nn.ReLU()
        input_size = list_of_features_sizes[0]

        # ColonTCN model
        self.stage1 = TCN(
            input_size=input_size,
            output_size=output_size,
            num_channels=list_of_features_sizes,
            kernel_size=kernel_size,
            num_of_convs=num_of_convs,
            dropout=dropout,
            residual=residual,
            last_layer=last_layer
        )


    def forward(self, x, mask=None):
        """
        Forward pass through the multi-stage TCN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, temporal_size, input_channels).
            mask (Tensor, optional): Mask tensor for selective temporal application.

        Returns:
            list of Tensor: Outputs from each stage (or single output if single-stage).
        """
        # Initial dimension reduction
        x = self.conv_first_layer_relu(self.conv_first_layer(x))

        # Apply ColonTCN
        out = self.stage1(x, mask)

        return [out]