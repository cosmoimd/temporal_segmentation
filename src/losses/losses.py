"""
Losses for models.
"""

# Import built-in modules
from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class TMSE(nn.Module):
    """
    Implements the Temporal MSE Loss Function proposed in "mstcn: Multi-Stage Temporal Convolutional Network for Action
    Segmentation" by Y. A. Farha et al. (CVPR 2019).

    The loss is computed as the mean of the squared difference between the logarithm of the softmax probability scores of
    adjacent predictions, with a threshold applied to clip the loss at a maximum value. The threshold and lambda_par
    hyperparameters can be used to control the scale of the loss.
    """

    def __init__(self, n_classes, threshold=4, lambda_par=1, ignore_index=999):
        """
        Initialize TMSE loss function.

        Args:
            n_classes (int or tuple of ints): number of classes for each output. If a tuple is provided, it is assumed
                                              that the model has multiple outputs and n_classes[i] corresponds to the
                                              number of classes for the i-th output.
            threshold (float, optional): threshold value to clamp the loss. Default is 4.
            lambda_par (float, optional): scaling factor for the loss. Default is 0.15.
            ignore_index (int, optional): index to ignore in the targets. Default is 999.
        """
        super().__init__()
        self.threshold = threshold ** 2
        self.lambda_par = lambda_par
        self.ignore_index = ignore_index
        self.n_classes = n_classes
        self.mse = nn.MSELoss(reduction='none')

    def loss_computation(self, output, targets, n_classes):
        # Reshape the target and output tensors into 1D arrays
        squashed_target = targets.view(-1)
        squashed_out = output.view(-1, n_classes)

        # Remove the ignore_index frames from the target tensor
        pred = squashed_out[torch.where(squashed_target != self.ignore_index)[0], :]

        # Compute loss
        loss = self.mse(F.log_softmax(pred[1:, :], dim=1), F.log_softmax(pred[:-1, :], dim=1))

        # Applies a threshold to the computed loss values to prevent the loss from getting too large.
        loss = torch.clamp(loss, min=0, max=self.threshold)

        # Computes the mean loss and normalizes it by the batch size.
        # This gives us the average loss per sample in the batch.
        return torch.mean(loss)

    def forward(self, output, targets):
        """
        Compute the forward pass of the TMSE loss function.

        Args:
            output: a tensor of shape (batch_size, sequence_length, num_classes) or a list of such tensors.
            targets: a tensor of shape (batch_size, sequence_length) containing the ground truth class labels.

        Returns:
            A scalar tensor representing the total TMSE loss across the batch.

        """
        total_loss = 0.

        if type(output) is list:
            # If the output is a list,
            # iterate through each output tensor and compute loss separately for each refinement.
            for out in output:
                total_loss += self.loss_computation(out[0], targets[0], self.n_classes[0])
                # If there are two refinements, compute the loss for the second refinement and add it to the total loss.
                if len(self.n_classes) == 2:
                    total_loss += self.loss_computation(out[1], targets[1], self.n_classes[1])
        else:
            # If the output is not a list, assume there is only one refinement and compute the loss accordingly.
            total_loss += self.loss_computation(output[0], targets[0], self.n_classes[0])
            # If there are two classification heads,
            # compute the loss for the second refinement and add it to the total loss.
            if len(self.n_classes) == 2:
                total_loss += self.loss_computation(output[1], targets[1], self.n_classes[1])

        # Multiply the total loss by the lambda parameter and return the result.
        return self.lambda_par * total_loss


class CE(nn.Module):
    """
        Weighted Cross Entropy Loss Function
    """

    def __init__(self, n_classes, class_weights, alpha=2, ignore_index=999, weight=None):
        """
        Initializes the CE class with the given parameters.

        Args:
        - n_classes (list): A list of integers representing the number of classes in each refinement stage.
        - class_weights (list): A list of weights for each class.
        - alpha (int): A hyperparameter used to balance the contribution of the two refinement stages.
        - ignore_index (int): An index to ignore during loss computation.

        Returns:

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction='none')
        self.n_classes = n_classes
        self.alpha = alpha
        self.weight = weight

        # Initializing the loss functions for each head
        if len(self.n_classes) > 1:
            self.loss_function = [
                nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights[0]).cuda(non_blocking=True),
                                    ignore_index=ignore_index,
                                    reduction="none"),
                nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights[1]).cuda(non_blocking=True),
                                    ignore_index=ignore_index,
                                    reduction="none")]
        else:
            self.loss_function = [nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights).cuda(non_blocking=True),
                ignore_index=ignore_index,
                reduction="none")]

    def forward(self, output, targets, per_frame_weights=None):
        """
        Computes the loss value between the predicted output and the ground truth targets.

        Args:
        - output (list): A list of tensors containing the predicted output for each refinement stage.
        - targets (list): A list of tensors containing the ground truth targets for each refinement stage.
        - per_frame_weights (list): A list of weights for each frame. This parameter is optional.

        Returns:
        - total_loss (tensor): A tensor representing the computed loss value.
        """
        total_loss = 0.

        if type(output) is list:
            for out in output:
                if len(self.n_classes) > 1:
                    for i in range(len(self.loss_function)):
                        model_output_i = out[i].reshape(-1, self.n_classes[i])
                        gt_values_i = targets[i].cuda(non_blocking=True).view(-1)

                        # Calculate the alpha value based on the refinement stage.
                        if i == 1:
                            alpha = self.alpha
                        else:
                            alpha = 1

                        if per_frame_weights is not None:
                            per_frame_weights_i = per_frame_weights[i].cuda(non_blocking=True).view(-1)
                            total_loss += alpha * torch.mean(per_frame_weights_i *
                                                             self.loss_function[i](model_output_i, gt_values_i))
                        else:
                            total_loss += alpha * torch.mean(self.loss_function[i](model_output_i, gt_values_i))
                else:

                    if per_frame_weights is not None:
                        per_frame_weights = per_frame_weights.cuda(non_blocking=True).view(-1)
                        total_loss += torch.mean(
                            per_frame_weights * self.loss_function[0](out.reshape(-1, self.n_classes[0]),
                                                                      targets.view(-1)))
                    else:
                        total_loss += torch.mean(
                            self.loss_function[0](out.reshape(-1, self.n_classes[0]), targets.view(-1)))
        else:
            for i in range(len(self.loss_function)):
                # Calculate the alpha value based on the refinement stage.
                if i == 1:
                    alpha = self.alpha
                else:
                    alpha = 1

                if per_frame_weights is not None:
                    per_frame_weights_i = per_frame_weights[i].cuda(non_blocking=True).view(-1)
                    total_loss += alpha * torch.mean(per_frame_weights_i *
                                                     self.loss_function[i](output.reshape(-1, self.n_classes[0]),
                                                                           targets.cuda(non_blocking=True).view(-1)))
                else:
                    total_loss += alpha * torch.mean(self.loss_function[i](output.reshape(-1, self.n_classes[0]),
                                                                           targets.cuda(non_blocking=True).view(-1)))

        return self.weight * total_loss


class FocalLoss(nn.Module):
  """
  Computes Focal Loss between target and output logits.
  """

  def __init__(self, class_weights=None, gamma=2, reduction='mean', weight=None):
    """
    Args:
      class_weights (optional): A list of weights for each class.
        If provided, the loss will be multiplied by these weights.
        Defaults to None.
      gamma (float, optional): Focusing parameter for the modulating factor.
        Defaults to 2.
      reduction (str, optional): Specifies the reduction to apply to the
        output: 'none' | 'mean' | 'sum'. Defaults to 'mean'.
      weight (torch.Tensor, optional): A manual rescaling weight
        if not using class_weights. Defaults to None.
    """
    super(FocalLoss, self).__init__()
    if class_weights is not None:
      self.class_weights = torch.FloatTensor(class_weights).cuda(non_blocking=True)
    else:
      self.class_weights = None
    self.gamma = gamma
    self.reduction = reduction
    self.weight = weight

  def forward(self, outputs, target, weights=None):
    total_loss = 0.
    for output in outputs:
      logpt = F.log_softmax(output, dim=2)
      pt = torch.exp(logpt)
      logpt = (1 - pt) ** self.gamma * logpt
      loss = F.nll_loss(logpt.view(-1, logpt.size(-1)), target.view(-1),
                        weight=self.class_weights,
                        ignore_index=999)
      total_loss += loss

    return self.weight * total_loss

