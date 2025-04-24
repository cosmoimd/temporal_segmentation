#!/usr/bin/env python
"""
Loss Fuctions used by Optmizers
"""

# Import necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class TMSE(nn.Module):
    """
    Temporal Mean Squared Error (TMSE) loss function for action segmentation.

    This loss penalizes differences in softmax log probabilities between adjacent frames,
    with an optional threshold to cap extreme values.
    """

    def __init__(self, n_classes, threshold=4, lambda_par=1, ignore_index=999):
        """
        Args:
            n_classes (int or tuple[int]): Number of classes for each output.
            threshold (float, optional): Maximum value to clamp the loss. Default is 4.
            lambda_par (float, optional): Scaling factor for the loss. Default is 1.
            ignore_index (int, optional): Index to ignore in the targets. Default is 999.
        """
        super().__init__()
        self.threshold = threshold ** 2
        self.lambda_par = lambda_par
        self.ignore_index = ignore_index
        self.n_classes = n_classes
        self.mse = nn.MSELoss(reduction='none')

    def loss_computation(self, output, targets, n_classes):
        """
        Computes the TMSE loss for a given set of predictions and targets.
        """
        squashed_target = targets.view(-1)
        squashed_out = output.view(-1, n_classes)
        pred = squashed_out[squashed_target != self.ignore_index]
        loss = self.mse(F.log_softmax(pred[1:], dim=1), F.log_softmax(pred[:-1], dim=1))
        loss = torch.clamp(loss, min=0, max=self.threshold)
        return torch.mean(loss)

    def forward(self, output, targets):
        """
        Forward pass to compute TMSE loss.
        """
        total_loss = 0.
        for out in output:
            total_loss += self.loss_computation(out[0], targets[0], self.n_classes)
        return self.lambda_par * total_loss


class CE(nn.Module):
    """
    Weighted Cross Entropy Loss Function with optional per-frame weighting.
    """

    def __init__(self, n_classes, class_weights, alpha=2, ignore_index=999, weight=None):
        """
        Args:
            n_classes (list[int]): Number of classes per refinement stage.
            class_weights (list[float]): Class weighting factors.
            alpha (int, optional): Balancing parameter for refinement stages. Default is 2.
            ignore_index (int, optional): Index to ignore in targets. Default is 999.
            weight (float, optional): Global loss scaling factor. Default is None.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.n_classes = n_classes
        self.alpha = alpha
        self.weight = weight
        self.loss_function = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).cuda(non_blocking=True),
            ignore_index=ignore_index,
            reduction="none"
        )

    def forward(self, output, targets):
        """
        Computes the cross-entropy loss.
        """
        total_loss = 0.
        for out in output:
            loss = self.loss_function(out.reshape(-1, self.n_classes), targets.view(-1))
            total_loss += torch.mean(loss)
        return self.weight * total_loss if self.weight else total_loss


class FocalLoss(nn.Module):
    """
    Computes Focal Loss to address class imbalance.
    """

    def __init__(self, class_weights=None, gamma=2, reduction='mean', weight=None):
        """
        Args:
            class_weights (list[float], optional): Class-wise weighting factors.
            gamma (float, optional): Focusing parameter. Default is 2.
            reduction (str, optional): Specifies reduction: 'none', 'mean', or 'sum'. Default is 'mean'.
            weight (float, optional): Global loss scaling factor. Default is None.
        """
        super().__init__()
        self.class_weights = torch.FloatTensor(class_weights).cuda(non_blocking=True) if class_weights else None
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, outputs, target, weights=None):
        """
        Computes focal loss across multiple outputs.
        """
        total_loss = 0.
        for output in outputs:
            logpt = F.log_softmax(output, dim=2)
            pt = torch.exp(logpt)
            logpt = (1 - pt) ** self.gamma * logpt
            loss = F.nll_loss(logpt.view(-1, logpt.size(-1)), target.view(-1),
                              weight=self.class_weights, ignore_index=999)
            total_loss += loss
        return self.weight * total_loss if self.weight else total_loss
