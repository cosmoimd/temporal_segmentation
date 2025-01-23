#!/usr/bin/env python

""" Train a model.

Usage: CUDA_VISIBLE_DEVICES=2 python3 training.py -parFile ymls/training_loc_mh.yml
"""

# Import built-in modules
from __future__ import print_function
import sys
import argparse
import yaml
import shutil
import os
import logging
import signal
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

class GracefulKiller:
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
        model (torch model): torch model to be saved
        data_loader (iterable): train or validation real_colon loader
        device (str): devices on which to load the real_colon
        n_classes (list): list of number of classes in the N-class classification problems
        loss_function (func): loss criterion adopted
        optimizer (object): torch optimizer
        lr_scheduler (scheduler): either None or scheduler object
        epoch_number (int): current epoch
        verbose (bool): whether to be verbose or not

    Returns:
        Return the avg classification accuracy of the model on the dataset.
    """
    total_loss = 0
    correct_prediction = 0
    number_of_predictions = 0
    num_updates = epoch_number * len(data_loader)
    for batch_idx, (matrix, target, _, _, mask, weights) in enumerate(data_loader):
        print("Epoch started")
        matrix, mask = matrix.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # video embeddings needs to have dimension (N, C, L) in order to be passed into CNN
        matrix = matrix.transpose(1, 2)

        if len(n_classes) == 2:
            target0, target1 = target[0].cuda(non_blocking=True), target[1].cuda(non_blocking=True)
            targets = [target0, target1]
        else:
            targets = target.cuda(non_blocking=True)

        if optimizer:
            optimizer.zero_grad()

        # Get model output prediction(s)
        output = model(matrix, mask)
        loss = 0.

        for lf in loss_function:
            if isinstance(lf, losses.TMSE):
                loss_current = lf(output, targets)
            else:
                loss_current = lf(output, targets, weights)
            loss += loss_current

        total_loss += loss.item()

        #  Update optimizer
        if optimizer:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()

        # Compute Classification Accuracy
        if compute_accuracy:
            # We consider the last prediction as the final one
            if type(output) is list:
                output = output[-1]

            # We compute accuracy as the % of correct temporal predictions
            if len(n_classes) == 2:
                _, predicted0 = torch.max(output[0].data, 2)
                _, predicted1 = torch.max(output[1].data, 2)
                iter_correct_prediction = ((predicted0 == target0).float() * mask).sum().item()
                iter_correct_prediction += ((predicted1 == target1).float() * mask).sum().item()
                iter_number_of_predictions = (999 != target0).float().sum().item() + (
                        999 != target1).float().sum().item()
                correct_prediction += iter_correct_prediction
                number_of_predictions += iter_number_of_predictions
            else:
                _, predicted = torch.max(output.data, 2)
                iter_correct_prediction = ((predicted == targets).float() * mask).sum().item()
                iter_number_of_predictions = (999 != targets).float().sum().item()
                correct_prediction += iter_correct_prediction
                number_of_predictions += iter_number_of_predictions

        # update learning rate scheduler, if used
        if lr_scheduler:
            lr_scheduler.step_update(num_updates=num_updates + batch_idx)

        if compute_accuracy and verbose:
            print('\nIteration Loss: {:.8f}  |  Accuracy: {:.4f}\n'.format(
                loss.item(), 100. * iter_correct_prediction / iter_number_of_predictions))

        if killer.kill_now:
            print("received kill!")
            break

    if compute_accuracy:
        accuracy = 100. * correct_prediction / number_of_predictions
        print('\nEpoch Loss: {:.8f}  |  Accuracy: {:.4f}\n'.format(total_loss, accuracy))
        return accuracy, total_loss
    else:
        return 0, 0


def main(args):
    # Read pas from input yml file
    if not os.path.exists(args.par_file):
        raise Exception("Parameter file %s does not exist" % (args.par_file))
    print("Parsing parfile: %s " % (args.par_file))
    with open(args.par_file, 'r') as stream:
        pars = yaml.safe_load(stream)
    print("Input pars: \n", pars)

    # Create training output folder
    pars["general"]["output_folder"] = core.resolvePathMacros(pars["general"]["output_folder"])
    os.makedirs(pars["general"]["output_folder"], exist_ok=True)

    # Create tensorboard and logging info
    os.makedirs(os.path.join(pars["general"]["output_folder"], "summary"), exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(pars["general"]["output_folder"], "summary"))
    logging.basicConfig(
        filename=os.path.join(pars["general"]["output_folder"], "experiment.log"),
        level=logging.INFO,
        format="%(asctime)s > %(message)s",
    )

    # Saving current git commit and config pars in output folder
    git.log_git_revision_to_file(pars["general"]["output_folder"])
    shutil.copy2(args.par_file, pars["general"]["output_folder"])

    # Save yml files used in the config file too
    ymls_in_the_cfg_file = core.find_key_in_dict(pars, ".yml")
    for path_to_yml in ymls_in_the_cfg_file:
        shutil.copy2(os.path.join(current_folder, path_to_yml),
                     pars["general"]["output_folder"])

    # Define model or reload it from a previous checkpoint
    if pars["model"]["model_type"] == "tcn":
        model = mstcn.MS_TCN(input_size=pars["model"]["input_size"],
                             output_size=pars["model"]["n_classes"],
                             num_channels=pars["model"]["channel_sizes"],
                             kernel_size=pars["model"]["kernel_size"],
                             residual=pars["model"]["residual"],
                             dropout=pars["model"]["dropout"],
                             conv_type=pars["model"]["conv_type"],
                             num_of_convs=pars["model"]["num_of_convs"],
                             conv_first=pars["model"]["conv_first"],
                             model_type=pars["model"]["model_type"],
                             last_layer=pars["model"]["last_layer"],
                             sigmoid_output=pars["model"]["sigmoid_output"],
                             )
    elif pars["model"]["model_type"] == "mstcn":
        model = mstcn.MS_TCN(input_size=pars["model"]["input_size"],
                             output_size=pars["model"]["n_classes"],
                             num_channels=pars["model"]["channel_sizes"],
                             kernel_size=pars["model"]["kernel_size"],
                             dropout=pars["model"]["dropout"],
                             residual=pars["model"]["residual"],
                             conv_type=pars["model"]["conv_type"],
                             num_of_convs=pars["model"]["num_of_convs"],
                             mstcn_input_size=pars["model"]["mstcn_input_size"],
                             mstcn_num_channels=pars["model"]["mstcn_channel_sizes"],
                             mstcn_kernel_size=pars["model"]["mstcn_kernel_size"],
                             mstcn_dropout=pars["model"]["mstcn_dropout"],
                             mstcn_residual=pars["model"]["mstcn_residual"],
                             mstcn_num_of_convs=pars["model"]["mstcn_num_of_convs"],
                             num_stages=pars["model"]["mstcn_num_stages"],
                             sigmoid_output=pars["model"]["sigmoid_output"],
                             model_type=pars["model"]["model_type"],
                             multiscale=pars["model"]["multiscale"],
                             dropout_before_last=pars["model"]["dropout_before_last"],
                             last_layer=pars["model"]["last_layer"],
                             conv_first=pars["model"]["conv_first"],
                             conv_before_last=pars["model"]["conv_before_last"])
    elif pars["model"]["model_type"] == "asformer":
        model = asformer.ASFormer(num_decoders=pars["model"]["num_decoders"],
                                  num_layers=pars["model"]["num_layers"],
                                  r1=pars["model"]["r1"],
                                  r2=pars["model"]["r2"],
                                  num_f_maps=pars["model"]["num_f_maps"],
                                  input_dim=pars["model"]["input_dim"],
                                  num_classes=pars["model"]["n_classes"][0],
                                  channel_masking_rate=pars["model"]["channel_masking_rate"])
    else:
        raise Exception("Wrong model_type!")

    # Reload model state, if specified
    if pars["model"]["model_path"] != "":
        if "tcn" in pars["model"]["model_type"]:
            model = mstcn_utils.reload_tcn_from_model_state_dict(model, pars["model"]["model_path"])
        else:
            model_checkpoint_path = core.resolvePathMacros(pars["model"]["model_path"])
            print("Reloading checkpoint from ", model_checkpoint_path)
            checkpoint = torch.load(model_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

    # Load model on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define real_colon loaders for training and validation
    compute_boundary_weights = pars["data_loader"]["compute_boundary_weights"] if "compute_boundary_weights" in pars[
        "data_loader"] else False
    train_dataset = embeddings_dataset.EmbeddingsDataset(pars["data_loader"]["train"],
                                                         temporal_augmentation=pars["data_loader"][
                                                             "temporal_augmentation"],
                                                         n_of_outputs=len(pars["model"]["n_classes"]),
                                                         prepare_dataset=pars["data_loader"]["prepare_dataset"],
                                                         compute_boundary_weights=compute_boundary_weights,
                                                         phase="training",
                                                         gaussian_noise=pars["data_loader"]["gaussian_noise"]
                                                         if "gaussian_noise" in pars["data_loader"] else None,
                                                         gaussian_noise_perc=pars["data_loader"]["gaussian_noise_perc"],
                                                         temp_folder=core.resolvePathMacros(
                                                             pars["data_loader"]["temp_folder"]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=pars["data_loader"]["batch_size"],
        num_workers=pars["data_loader"]["num_workers"],
        collate_fn=embeddings_dataset.custom_collate_batch_training,
        shuffle=True,
        pin_memory=False)
    val_dataset = embeddings_dataset.EmbeddingsDataset(pars["data_loader"]["valid"],
                                                       n_of_outputs=len(pars["model"]["n_classes"]),
                                                       phase="validation",
                                                       prepare_dataset=pars["data_loader"]["prepare_dataset"],
                                                       compute_boundary_weights=compute_boundary_weights,
                                                       temp_folder=core.resolvePathMacros(
                                                           pars["data_loader"]["temp_folder"])
                                                       )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=pars["data_loader"]["num_workers"],
        collate_fn=embeddings_dataset.custom_collate_batch_training,
        shuffle=False,
        pin_memory=False)

    # Set up optimizer and lr scheduler
    optimizer = build_optimizers.build_optimizer(model, pars['optimizer'])
    if pars['optimizer']['scheduler_name'] != "":
        lr_scheduler = learning_rate_schedulers.build_scheduler(pars['optimizer']['scheduler_name'],
                                                                optimizer,
                                                                min_lr=pars['optimizer']['min_lr'],
                                                                warmup_lr=pars['optimizer']['warmup_lr'],
                                                                decay_rate=pars['optimizer']['decay_rate'],
                                                                epochs=pars['optimizer']['n_epochs'],
                                                                warmup_epochs=pars['optimizer']['warmup_epochs'],
                                                                decay_epochs=pars['optimizer']['decay_epochs'],
                                                                cycle_limit=pars['optimizer']['cycle_limit'],
                                                                n_iter_per_epoch=len(train_loader))
    else:
        lr_scheduler = None

    if pars["optimizer"]["class_weights"] == "compute":
        class_weights = train_dataset.compute_weights()
    else:
        class_weights = pars["optimizer"]["class_weights"]

    # Set up loss function
    loss_function = []
    if "CE" in pars["optimizer"]["losses"]:
        loss_function += [losses.CE(pars["model"]["n_classes"], class_weights,
                                    alpha=pars["optimizer"]["alpha"],
                                    weight=pars["optimizer"]["ce_weight"])]
    if "Focal" in pars["optimizer"]["losses"]:
        loss_function += [losses.FocalLoss(class_weights=class_weights,
                                           weight=pars["optimizer"]["focal_weight"])]
    if "Dice" in pars["optimizer"]["losses"]:
        loss_function += [losses.DiceLoss(weight=pars["optimizer"]["dice_weight"])]
    if "TMSE" in pars["optimizer"]["losses"]:
        loss_function += [losses.TMSE(pars["model"]["n_classes"])]

    # Print Model Summary
    print(model)

    killer = GracefulKiller()

    # Training loop: loops over epochs
    max_validation_accuracy = 0.
    max_validation_accuracy_after500 = 0.
    max_validation_accuracy_after1000 = 0.
    for epoch_number in tqdm(range(pars["optimizer"]["n_epochs"]), desc="epoch"):
        print("Epoch: %d. \n" % epoch_number)

        if killer.kill_now:
            print("received kill!")
            break

        if epoch_number % pars["optimizer"]["save_checkpoint_N_epochs"] == 0:
            compute_accuracy = True
        else:
            compute_accuracy = False

        # Fit train real_colon to the model and update its weights in train mode
        print("Training.")
        model.train()
        train_accuracy, train_loss = epoch(killer=killer,
                                           model=model,
                                           data_loader=train_loader,
                                           n_classes=pars["model"]["n_classes"],
                                           loss_function=loss_function,
                                           optimizer=optimizer,
                                           epoch_number=epoch_number,
                                           lr_scheduler=lr_scheduler,
                                           compute_accuracy=compute_accuracy)

        # Run validation and save and tensorboard
        if epoch_number % pars["optimizer"]["save_checkpoint_N_epochs"] == 0:
            print("Validation.")
            model.eval()
            with torch.no_grad():
                valid_accuracy, valid_loss = epoch(killer=killer,
                                                   model=model,
                                                   data_loader=val_loader,
                                                   n_classes=pars["model"]["n_classes"],
                                                   loss_function=loss_function,
                                                   optimizer=None,
                                                   compute_accuracy=compute_accuracy)

            # save checkpoint
            output_model_path = os.path.join(pars["general"]["output_folder"], f"epoch-{epoch_number}_ta_"
                                                                               f"{round(train_accuracy, 3)}_va_"
                                                                               f"{round(valid_accuracy, 3)}.pth")
            torch.save({
                'epoch': epoch_number,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_model_path)

            # save checkpoint with max valid acc
            if valid_accuracy > max_validation_accuracy:
                torch.save({
                    'epoch': epoch_number,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(pars["general"]["output_folder"], "best_val_acc.pth"))
                max_validation_accuracy = valid_accuracy

            # save checkpoint with max valid acc
            if valid_accuracy > max_validation_accuracy and epoch_number > 500:
                torch.save({
                    'epoch': epoch_number,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(pars["general"]["output_folder"], "best_val_acc_after_500_epochs.pth"))
                max_validation_accuracy_after500 = valid_accuracy

            if valid_accuracy > max_validation_accuracy and epoch_number > 1000:
                torch.save({
                    'epoch': epoch_number,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(pars["general"]["output_folder"], "best_val_acc_after_1000_epochs.pth"))
                max_validation_accuracy_after1000 = valid_accuracy

            # tensorboard writing
            summary_writer.add_scalars(
                "lr", {"lr": optimizer.param_groups[0]["lr"]}, epoch_number
            )
            summary_writer.add_scalars(
                "loss", {"val": valid_loss, "train": train_loss}, epoch_number
            )
            summary_writer.add_scalars(
                "accuracy", {"val": valid_accuracy, "train": train_accuracy}, epoch_number
            )

            # log writing
            opt_lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch {epoch_number}, lr: {opt_lr:.8f}, "
                f"train loss: {train_loss:.3f}, train acc: {train_accuracy:.3f}, "
                f"valid loss: {valid_loss:.3f}, valid acc: {valid_accuracy:.3f}, "
            )

    # save last/end of training checkpoint
    torch.save({
        'epoch': epoch_number,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(pars["general"]["output_folder"], "last_checkpoint.pth"))

    summary_writer.close()
    print("Training has ended.")


if __name__ == '__main__':
    # Parse input and start main
    parser = argparse.ArgumentParser(description='Train a TCN model on one or more datasets')
    parser.add_argument('-parFile', action='store', dest='par_file',
                        help='path to the parameter file', required=True)
    args = parser.parse_args()

    main(args)
