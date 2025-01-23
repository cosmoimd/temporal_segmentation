"""
Define single-stage and multi-stage Temporal Convolutional Network (TCN) architectures.
"""
import torch
import torch.nn as nn
from src.models import layers as model_layers


class TCN(nn.Module):
    """
    Temporal convolutional network (TCN) and ColonTCN video temporal segmentation.
    """

    def __init__(self, input_size, output_size=None, num_channels=None, conv_type="causal", num_of_convs=2,
                 kernel_size=2, dropout=0.2, residual=True, last_layer="linear", conv_before_last=False,
                 dropout_before_last=False, sigmoid_output=False):
        """
        Initializes a TemporalConvNet model.

        Args:
            input_size (int): Number of input channels.
            output_size (int or None): Number of output classes. If None, no output layer is applied.
            num_channels (list of int): List of output channels for each temporal block.
            conv_type (str): Convolution type, "causal" or "non-causal".
            num_of_convs (int): Number of convolutions per temporal block.
            kernel_size (int): Kernel size for each convolution.
            dropout (float): Dropout rate after each convolutional layer.
            residual (bool): Use residual connections within each temporal block.
            last_layer (str): Type of last layer, "linear" or "conv".
            conv_before_last (bool): Apply an additional convolution before the final layer.
            dropout_before_last (bool): Apply dropout before the final layer.
            sigmoid_output (bool): Apply sigmoid activation to the output.
        """
        super(TCN, self).__init__()
        self.conv_before_last = conv_before_last
        self.last_layer_type = last_layer
        self.sigmoid_output = sigmoid_output
        self.output_size = output_size
        self.dropout_before_last = dropout_before_last

        # Build temporal blocks
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            padding = int((kernel_size - 1) * dilation_size / 2) if conv_type != "causal" else int(
                (kernel_size - 1) * dilation_size)

            layers.append(model_layers.TemporalBlock(
                in_channels, out_channels, kernel_size, conv_type=conv_type,
                num_of_convs=num_of_convs, stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout, residual=residual
            ))

        self.network = nn.Sequential(*layers)
        num_channels_final = num_channels[-1]

        # Optional convolution layer before final output
        if self.conv_before_last:
            self.conv_before_last_layer = nn.Conv1d(num_channels_final, num_channels_final // 4, 1)
            self.relu = nn.ReLU()
            num_channels_final //= 4

        # Define final output layer
        if output_size is not None:
            self.last_layer = (nn.Linear(num_channels_final, output_size) if last_layer == "linear"
                               else nn.Conv1d(num_channels_final, output_size, 1))
            if sigmoid_output:
                self.sig = nn.Sigmoid()

            # Optional dropout layer before the final output layer
            if self.dropout_before_last:
                self.layer_dropout_before_last = nn.Dropout(dropout)

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

        # Apply optional conv layer before the final output
        if self.conv_before_last:
            output = self.relu(self.conv_before_last_layer(output))

        # Apply final output layer if output_size is defined
        if self.output_size is not None:
            if self.dropout_before_last:
                output = self.layer_dropout_before_last(output)
            net_output = (self.last_layer(output.permute(0, 2, 1)).float() if self.last_layer_type == "linear"
                          else self.last_layer(output).float().permute(0, 2, 1))

            # Apply mask and sigmoid activation if specified
            if mask is not None:
                return self.sig(net_output) * mask.unsqueeze(2) if self.sigmoid_output else net_output * mask.unsqueeze(
                    2)
            else:
                return self.sig(net_output) if self.sigmoid_output else net_output
        else:
            # If no output layer, return TCN feature outputs directly
            return output

class MS_TCN(nn.Module):
    """
    Multi-stage TCN and ColonTCN (mstcn or MS-ColonTCN)
    """

    def __init__(self, input_size, output_size, list_of_features_sizes, kernel_size, conv_type="causal", num_of_convs=2,
                 mstcn_num_stages=0, dropout=1.0, mstcn_input_size=None, mstcn_list_of_features_sizes=None, mstcn_kernel_size=None,
                 mstcn_num_of_convs=None, mstcn_dropout=1.0, residual=True, conv_first=False, mstcn_residual=True,
                 sigmoid_output=False, last_layer="linear", conv_before_last=False, dropout_before_last=False):
        """
        Initializes a multi-stage TCN (mstcn) with optional refinement stages.

        Args:
            input_size (int): Number of input channels.
            output_size (int or list of int): Number of output channels or classes.
            list_of_features_sizes (list of int): Number of channels in the base TCN layers.
            kernel_size (int): Kernel size for convolutions.
            conv_type (str): Type of convolution, "causal" or "non-causal".
            num_of_convs (int): Number of convolutions per block in base TCN.
            num_stages (int): Number of additional refinement stages.
            dropout (float): Dropout rate for the TCN layers.
            mstcn_input_size (int): Input size for multi-stage TCN stages.
            mstcn_list_of_features_sizes (list of int): Number of channels for each level.
            mstcn_kernel_size (int): Kernel size for multi-stage layers.
            mstcn_num_of_convs (int): Number of convolutions per block in each stage.
            mstcn_dropout (float): Dropout rate for multi-stage layers.
            residual (bool): Whether to use residual connections within base TCN blocks.
            conv_first (bool): Apply a 1D convolution before base TCN layers.
            mstcn_residual (bool): Use residual connections within multi-stage TCN blocks.
            sigmoid_output (bool): Apply sigmoid activation on final output.
            last_layer (str): Type of final layer ("linear" or "conv").
            conv_before_last (bool): Apply a 1D convolution before the final output.
            dropout_before_last (bool): Apply dropout before the final layer.
        """
        super(MS_TCN, self).__init__()
        self.num_stages = mstcn_num_stages
        self.conv_first = conv_first

        # Optional initial convolution layer to reduce input dimensionality
        if self.conv_first:
            self.conv_first_layer = nn.Conv1d(input_size, list_of_features_sizes[0], 1)
            self.conv_first_layer_relu = nn.ReLU()
            input_size = list_of_features_sizes[0]

        # Initial (base) TCN layer
        self.stage1 = TCN(
            input_size=input_size,
            output_size=output_size,
            num_channels=list_of_features_sizes,
            kernel_size=kernel_size,
            conv_type=conv_type,
            num_of_convs=num_of_convs,
            dropout=dropout,
            residual=residual,
            last_layer=last_layer,
            conv_before_last=conv_before_last,
            dropout_before_last=dropout_before_last,
            sigmoid_output=sigmoid_output
        )

        # Multi-stage refinement layers, if specified
        if self.num_stages > 0:
            self.stages = nn.ModuleList([
                TCN(
                    input_size=mstcn_input_size,
                    output_size=output_size,
                    num_channels=mstcn_list_of_features_sizes,
                    kernel_size=mstcn_kernel_size,
                    conv_type=conv_type,
                    num_of_convs=mstcn_num_of_convs,
                    dropout=mstcn_dropout,
                    residual=mstcn_residual,
                    last_layer=last_layer,
                    conv_before_last=conv_before_last,
                    dropout_before_last=dropout_before_last,
                    sigmoid_output=sigmoid_output
                ) for _ in range(self.num_stages)
            ])

    def forward(self, x, mask=None):
        """
        Forward pass through the multi-stage TCN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, temporal_size, input_channels).
            mask (Tensor, optional): Mask tensor for selective temporal application.

        Returns:
            list of Tensor: Outputs from each stage (or single output if single-stage).
        """
        # Initial dimension reduction, if specified
        if self.conv_first:
            x = self.conv_first_layer_relu(self.conv_first_layer(x))

        # First stage (base TCN)
        out = self.stage1(x, mask)
        outputs = [out]

        # Additional refinement stages (if multi-stage)
        if self.num_stages > 0:
            for stage in self.stages:
                x = out.transpose(1, 2)  # Transpose to match input requirements
                out = stage(x, mask)
                outputs.append(out)

        return outputs  # List of outputs from each stage (base and refinements if any)