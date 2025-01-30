#!/usr/bin/env python

"""
Train a model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/training.py -parFile ymls/training/colontcn_4fold/training_colontcn_4fold_fold1.yml
"""

import sys
import argparse
import yaml
import os
import logging
import signal
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

from src.models.factory import ModelFactory
from src.optimizers.builders import build_optimizer, build_scheduler
from src.optimizers import losses
from src.data_loader.embeddings_dataset import EmbeddingsDataset, custom_collate_batch_training


class GracefulKiller:
    """Handle graceful termination of the training process."""
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


def epoch(killer, model, data_loader, n_classes, loss_function, optimizer=None, lr_scheduler=None,
          epoch_number=0, verbose=False, compute_accuracy=True):
    """
    Train or Validate a model on a single epoch.

    Args:
        killer (GracefulKiller): Signal handler for interruptions.
        model (torch.nn.Module): Model to be trained or validated.
        data_loader (DataLoader): Training or validation data loader.
        n_classes (list): Number of classes in the classification problem.
        loss_function (list): List of loss functions used for training.
        optimizer (torch.optim.Optimizer, optional): Optimizer for model training. Defaults to None.
        lr_scheduler (Scheduler, optional): Learning rate scheduler. Defaults to None.
        epoch_number (int, optional): Current epoch number. Defaults to 0.
        verbose (bool, optional): Whether to print detailed logs. Defaults to False.
        compute_accuracy (bool, optional): Whether to compute accuracy. Defaults to True.

    Returns:
        tuple: Tuple containing accuracy and total loss.
    """
    total_loss, correct_prediction, number_of_predictions = 0, 0, 0
    num_updates = epoch_number * len(data_loader)

    for batch_idx, (matrix, target, _, _, mask, weights) in enumerate(data_loader):
        if killer.kill_now:
            print("Training interrupted!")
            break

        matrix, mask = matrix.cuda(non_blocking=True), mask.cuda(non_blocking=True)
        matrix = matrix.transpose(1, 2)  # Adjust input dimensions
        targets = [t.cuda(non_blocking=True) for t in target] if isinstance(target, list) else target.cuda(
            non_blocking=True)

        if optimizer:
            optimizer.zero_grad()

        output = model(matrix, mask)
        loss = sum(lf(output, targets) if isinstance(lf, losses.TMSE) else lf(output, targets, weights) for lf in
                   loss_function)
        total_loss += loss.item()

        if optimizer:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()

        if compute_accuracy:
            output = output[-1] if isinstance(output, list) else output
            _, predicted = torch.max(output.data, 2)
            mask_float = mask.float()
            iter_correct_prediction = ((predicted == targets).float() * mask_float).sum().item()
            iter_number_of_predictions = (999 != targets).float().sum().item()
            correct_prediction += iter_correct_prediction
            number_of_predictions += iter_number_of_predictions

        if lr_scheduler:
            lr_scheduler.step_update(num_updates=num_updates + batch_idx)

    accuracy = 100. * correct_prediction / number_of_predictions if compute_accuracy else 0
    return accuracy, total_loss


def main(args):
    """Main function to train the model."""
    if not os.path.exists(args.par_file):
        raise FileNotFoundError(f"Parameter file {args.par_file} does not exist")

    with open(args.par_file, 'r') as stream:
        pars = yaml.safe_load(stream)

    os.makedirs(pars["general"]["output_folder"], exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(pars["general"]["output_folder"], "summary"))

    logging.basicConfig(
        filename=os.path.join(pars["general"]["output_folder"], "experiment.log"),
        level=logging.INFO,
        format="%(asctime)s > %(message)s",
    )

    model_factory = ModelFactory(pars["model"])
    model = model_factory.create_model()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = build_optimizer(model, pars['optimizer'])
    lr_scheduler = build_scheduler(pars['optimizer']['scheduler_name'], optimizer, **pars['optimizer']) if \
    pars['optimizer']['scheduler_name'] else None

    loss_function = [losses.CE(pars["model"]["output_size"], pars["optimizer"]["class_weights"])]

    train_dataset = EmbeddingsDataset(pars["data_loader"]["train"], phase="training")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=pars["data_loader"]["batch_size"],
                                               num_workers=pars["data_loader"]["num_workers"], shuffle=True)
    val_dataset = EmbeddingsDataset(pars["data_loader"]["valid"], phase="validation")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=pars["data_loader"]["num_workers"],
                                             shuffle=False)

    killer = GracefulKiller()

    for epoch_number in tqdm(range(pars["optimizer"]["n_epochs"]), desc="Training Progress"):
        if killer.kill_now:
            print("Training terminated early.")
            break

        print(f"Epoch {epoch_number}")
        model.train()
        train_accuracy, train_loss = epoch(killer, model, train_loader, pars["model"]["output_size"], loss_function,
                                           optimizer, lr_scheduler, epoch_number)

        if epoch_number % pars["optimizer"]["save_checkpoint_N_epochs"] == 0:
            model.eval()
            with torch.no_grad():
                valid_accuracy, valid_loss = epoch(killer, model, val_loader, pars["model"]["output_size"],
                                                   loss_function)
            torch.save({'epoch': epoch_number, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(pars["general"]["output_folder"], f"checkpoint_{epoch_number}.pth"))

    summary_writer.close()
    print("Training completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-parFile', required=True, help='Path to the parameter file')
    args = parser.parse_args()
    main(args)
