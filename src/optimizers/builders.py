#!/usr/bin/env python
"""
Build Learning Rate Schedulers and Optimizers.
"""

# Import necessary modules
from __future__ import print_function
import torch
import torch.optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def build_optimizer(model: torch.nn.Module, pars: dict) -> torch.optim.Optimizer:
    """
    Creates and returns an optimizer based on provided parameters.

    Args:
        model (torch.nn.Module): The model for which the optimizer is created.
        pars (dict): Dictionary containing optimizer parameters from the YAML file.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    Raises:
        AssertionError: If optimizer type is not supported.
    """
    assert pars['type'] in ['sgd', 'adam', 'adamw', 'RMSprop'], "Unsupported optimizer type"

    optimizers = {
        'sgd': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam
    }

    kwargs = {
        'lr': pars['start_lr'],
        'weight_decay': float(pars.get('weight_decay', 0))
    }

    if pars['type'] == 'sgd':
        kwargs['momentum'] = pars.get('momentum', 0.9)

    return optimizers[pars['type']](model.parameters(), **kwargs)


def build_scheduler(
        scheduler_name: str, optimizer: torch.optim.Optimizer, min_lr: float, warmup_lr: float,
        decay_rate: float, epochs: int, warmup_epochs: int, decay_epochs: int, n_iter_per_epoch: int,
        cycle_limit: int = 1, t_mult: int = 1
) -> Scheduler:
    """
    Creates and returns a learning rate scheduler based on provided parameters.

    Args:
        scheduler_name (str): One of ['cosine', 'torch_cosine', 'linear', 'step'].
        optimizer (torch.optim.Optimizer): Optimizer instance for which the scheduler is created.
        min_lr (float): Minimum learning rate.
        warmup_lr (float): Initial learning rate for warmup.
        decay_rate (float): Decay rate for step scheduler.
        epochs (int): Total number of epochs in training.
        warmup_epochs (int): Number of warmup epochs.
        decay_epochs (int): Epoch interval for learning rate decay.
        n_iter_per_epoch (int): Number of iterations per epoch.
        cycle_limit (int, optional): Number of cycles for cosine scheduler. Defaults to 1.
        t_mult (int, optional): Factor for increasing cycle length. Defaults to 1.

    Returns:
        Scheduler: Configured learning rate scheduler instance.
    """
    num_steps, warmup_steps, decay_steps = epochs * n_iter_per_epoch, warmup_epochs * n_iter_per_epoch, decay_epochs * n_iter_per_epoch

    schedulers = {
        'torch_cosine': CosineAnnealingWarmRestarts(optimizer, T_0=epochs // cycle_limit, T_mult=t_mult,
                                                    eta_min=min_lr),
        'cosine': CosineLRScheduler(optimizer, t_initial=num_steps, lr_min=min_lr, warmup_lr_init=warmup_lr,
                                    warmup_t=warmup_steps, cycle_limit=cycle_limit, t_in_epochs=False),
        'linear': LinearLRScheduler(optimizer, t_initial=num_steps, lr_min_rate=0.01, warmup_lr_init=warmup_lr,
                                    warmup_t=warmup_steps, t_in_epochs=False),
        'step': StepLRScheduler(optimizer, decay_t=decay_steps, decay_rate=decay_rate, warmup_lr_init=warmup_lr,
                                warmup_t=warmup_steps, t_in_epochs=False)
    }

    return schedulers.get(scheduler_name, None)


class LinearLRScheduler(Scheduler):
    """
    Implements a Linear Learning Rate Scheduler.
    """

    def __init__(
            self, optimizer: torch.optim.Optimizer, t_initial: int, lr_min_rate: float, warmup_t: int = 0,
            warmup_lr_init: float = 0., t_in_epochs: bool = True, noise_range_t=None,
            noise_pct: float = 0.67, noise_std: float = 1.0, noise_seed: int = 42, initialize: bool = True
    ) -> None:
        """
        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance.
            t_initial (int): Total steps computed as epochs * n_iter_per_epoch.
            lr_min_rate (float): Minimum learning rate as a fraction of base learning rate.
            warmup_t (int, optional): Warmup steps. Defaults to 0.
            warmup_lr_init (float, optional): Initial learning rate for warmup. Defaults to 0.
            t_in_epochs (bool, optional): Whether scheduler works in epochs. Defaults to True.
            noise_range_t (optional): Noise range threshold.
            noise_pct (float, optional): Noise percent limit. Defaults to 0.67.
            noise_std (float, optional): Noise standard deviation. Defaults to 1.0.
            noise_seed (int, optional): Seed for noise randomization. Defaults to 42.
            initialize (bool, optional): If True, initializes optimizer parameter groups. Defaults to True.
        """
        super().__init__(optimizer, param_group_field="lr", noise_range_t=noise_range_t, noise_pct=noise_pct,
                         noise_std=noise_std, noise_seed=noise_seed, initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t if self.warmup_t else 1 for v in self.base_values]
        if self.warmup_t:
            super().update_groups(self.warmup_lr_init)

    def _get_lr(self, t: int) -> list:
        """
        Computes learning rate for a given training step.

        Args:
            t (int): Current training step.
        Returns:
            list: Learning rates for each parameter group.
        """
        if t < self.warmup_t:
            return [self.warmup_lr_init + t * s for s in self.warmup_steps]
        t -= self.warmup_t
        total_t = self.t_initial - self.warmup_t
        return [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]

    def get_epoch_values(self, epoch: int):
        return self._get_lr(epoch) if self.t_in_epochs else None

    def get_update_values(self, num_updates: int):
        return self._get_lr(num_updates) if not self.t_in_epochs else None