#!/usr/bin/env python

""" Compute video classification frame-wise metrics.

Usage:
        python3 testing.py -parFile real_colon/testing_phase.yml
"""
import numpy as np
import sys
import argparse
import yaml
import os
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

matplotlib.use('Agg')

def abs_perc_difference(predictions, ground_truths, withdrawal_labels=set([2, 3, 4, 5, 6, 7, 8])):
    # Filter out the relevant withdrawal labels
    filtered_preds = []
    filtered_gts = []
    for pred, gt in zip(predictions, ground_truths):
        if pred in withdrawal_labels:
            filtered_preds.append(pred)

        if gt in withdrawal_labels:
            filtered_gts.append(gt)

    mape = abs((len(filtered_gts) - len(filtered_preds)) / len(filtered_gts)) if len(filtered_gts) > 0 else 0
    return mape


def compute_weighted_jaccard_index(predictions, ground_truths, num_classes=9):
    # Initialize counts for intersection, union, and frequency for each class
    intersection = [0] * num_classes
    union = [0] * num_classes
    frequency = [0] * num_classes  # Track the number of instances for each class

    # Iterate through each frame
    for pred, gt in zip(predictions, ground_truths):
        # Update class frequency
        frequency[gt] += 1

        if pred == gt:
            # Update intersection for correct predictions
            intersection[pred] += 1

        # Update union
        union[pred] += 1
        if pred != gt:
            union[gt] += 1

    # Calculate weighted Jaccard index for each class
    weighted_jaccard_sum = 0
    total_instances = sum(frequency)  # Total number of instances across all classes
    for i in range(num_classes):
        class_jaccard = intersection[i] / union[i] if union[i] else 0
        weighted_jaccard_sum += class_jaccard * frequency[i]

    # Compute the class-weighted average Jaccard score
    weighted_avg_jaccard = weighted_jaccard_sum / total_instances if total_instances else 0

    return weighted_avg_jaccard


def compute_jaccard_index(predictions, ground_truths, num_classes=9):
    # Initialize counts for intersection and union for each class
    intersection = [0] * num_classes
    union = [0] * num_classes

    # Iterate through each frame
    for pred, gt in zip(predictions, ground_truths):
        if pred == gt:
            # Update intersection for correct predictions
            intersection[pred] += 1

        # Update union
        union[pred] += 1
        if pred != gt:
            union[gt] += 1

    # Calculate Jaccard index for each class
    jaccard_per_class = [intersection[i] / union[i] if union[i] else 0 for i in range(num_classes)]
    avg_jaccard = sum(jaccard_per_class) / num_classes

    return jaccard_per_class, avg_jaccard


def testing(pars):
    """
    Computes classification testing metrics on tcn inference output.

    Args:
        pars (dict): a parameters dictionary
    """
    print("TCN testing pars: \n", pars)
    pars["general"]["output_folder"] = pars["general"]["output_folder"]
    os.makedirs(pars["general"]["output_folder"], exist_ok=True)

    total_gts = []  # overall list of all the frame gts from all the videos
    total_predictions = []  # overall list of all the predictions from all the videos

    # loop over datasets, one output folder for each datasets
    for dataset_dict in pars["data_loader"]["valid"]:
        # create dataset output folder and labs subfolder
        current_output_folder = os.path.join(pars["general"]["output_folder"], dataset_dict["dataset_name"])
        os.makedirs(current_output_folder, exist_ok=True)

        # List of video filenames to study
        with open(dataset_dict["video_list"], "r") as f:
            video_names = [line.rstrip() for line in f]

        # Loop over GTs, given a GT loop over videos in that dataset
        abs_perc_difference_w = []
        current_output_folder = os.path.join(pars["general"]["output_folder"], dataset_dict["dataset_name"], dataset_dict['gt_name'])
        os.makedirs(current_output_folder, exist_ok=True)

        # Run testing, video by video
        # Collect GTs and predictions to compute dataset metrics
        tot_correct = 0
        tot_counter = 0
        gts = []  # overall list of all the frame gts from all the dataset videos
        predictions = []  # overall list of all the predictions from all the dataset videos

        for video_name in video_names:
            cvn = video_name + ".pkl"

            # Load model output prediction for video_name
            op = os.path.join(pars["general"]["inference_output"], cvn)
            with open(op, "rb") as fp:
                inference_output = pickle.load(fp)
            print("Computing testing metrics and plots for video: ", video_name)

            # Retrieve computed time series predictions
            model_output = inference_output["model_output"][0]

            pred = list(np.argmax(model_output, axis=2).flatten())
            list_of_frame_names = inference_output["list_of_frame_names"]

            # process dataset in csv (input of classification_rnn)
            csv_path = os.path.join(dataset_dict["csv_path"], cvn[:-4] + ".csv")
            csv_file = pd.read_csv(csv_path)
            list_of_frame_names = csv_file[csv_file["frame_filename"].isin(list_of_frame_names)][
                "frame_filename"].to_list()
            gt = csv_file[csv_file["frame_filename"].isin(list_of_frame_names)][dataset_dict['gt_name']].to_list()
            gt = [dataset_dict["str_to_idx"][g] for g in gt]

            # Remove frames at class 999
            pred = [p for i, p in enumerate(pred) if gt[i] != 999]
            list_of_frame_names = [p for i, p in enumerate(list_of_frame_names) if gt[i] != 999]
            gt = [g for i, g in enumerate(gt) if g != 999]

            abs_perc_difference_w.append(abs_perc_difference(pred, gt))

            # Update total lists
            gts += gt
            predictions += pred

            # Update num and den for total accuracy
            tot_correct += sum(x == y for x, y in zip(pred, gt))
            tot_counter += len(pred)

        # Classification metrics and confusion matrix for total gt vs pred time series
        classification_rep_save = classification_report(gts, predictions, output_dict=True)
        jaccard_per_class, avg_jaccard = compute_jaccard_index(predictions, gts,
                                                                    num_classes=pars["model"]["output_size"])
        weighted_jaccard_per_class = compute_weighted_jaccard_index(predictions, gts,
                                                                    num_classes=pars["model"]["output_size"])
        df_report = pd.DataFrame(classification_rep_save).transpose()
        df_report["jaccard_per_class"] = (jaccard_per_class + [avg_jaccard] *
                                          (df_report.shape[0] - pars["model"]["output_size"]))
        df_report["weighted_jaccard_per_class"] = (jaccard_per_class +
                                                   [weighted_jaccard_per_class] *
                                                   (df_report.shape[0] - pars["model"]["output_size"]))

        # String to index items
        str_to_idx_items = dataset_dict['str_to_idx'].items()

        # Save classification report using 1 decimal places (88.1) and get class names from str_to_idx
        df_report["precision"] *= 100
        df_report["recall"] *= 100
        df_report["f1-score"] *= 100
        df_report["jaccard_per_class"] *= 100
        df_report["weighted_jaccard_per_class"] *= 100
        df_report = np.round(df_report, decimals=1)
        df_report = df_report.rename(index=dict((str(v), k) for k, v in str_to_idx_items))
        df_report.to_csv(os.path.join(current_output_folder, "gt_vs_pred_perFrame.csv"))

        # Save confusion matrix with correct namings
        cm_images = confusion_matrix(gts, predictions)
        df_report = pd.DataFrame(cm_images)
        df_report = df_report.rename(index=dict((v, k) for k, v in str_to_idx_items))
        df_report = df_report.rename(columns=dict((v, k) for k, v in str_to_idx_items))
        df_report.to_csv(os.path.join(current_output_folder, "gt_vs_pred_cm.csv"))

        total_gts += gts
        total_predictions += predictions

        # Classification metrics and confusion matrix aggregated over all datasets
        classification_rep_save = classification_report(total_gts, total_predictions, output_dict=True)
        df_report = pd.DataFrame(classification_rep_save).transpose()
        str_to_idx_items = dataset_dict['str_to_idx'].items()
        df_report["precision"] *= 100
        df_report["recall"] *= 100
        df_report["f1-score"] *= 100
        jaccard_per_class, avg_jaccard = compute_jaccard_index(total_predictions, total_gts, num_classes=pars["model"]["output_size"])
        weighted_jaccard_per_class = compute_weighted_jaccard_index(total_predictions, total_gts, num_classes=pars["model"]["output_size"])
        df_report["abs_perc_difference_w"] = [np.mean(abs_perc_difference_w) * 100] * df_report.shape[0]
        df_report["jaccard_per_class"] = (jaccard_per_class + [avg_jaccard] *
                                          (df_report.shape[0] - pars["model"]["output_size"]))
        df_report["weighted_jaccard_per_class"] = (jaccard_per_class +
                                                   [weighted_jaccard_per_class] *
                                                   (df_report.shape[0] - pars["model"]["output_size"]))
        df_report["jaccard_per_class"] *= 100
        df_report["weighted_jaccard_per_class"] *= 100
        df_report = np.round(df_report, decimals=1)
        df_report = df_report.rename(index=dict((str(v), k) for k, v in str_to_idx_items))
        report_path = os.path.join(pars["general"]["output_folder"], str(dataset_dict['gt_name']) + "gt_vs_pred_perFrame.csv")
        df_report.to_csv(report_path)
        cm_images = confusion_matrix(total_gts, total_predictions)
        df_report = pd.DataFrame(cm_images)
        df_report = df_report.rename(index=dict((v, k) for k, v in str_to_idx_items))
        df_report = df_report.rename(columns=dict((v, k) for k, v in str_to_idx_items))
        df_report.to_csv(os.path.join(pars["general"]["output_folder"], str(dataset_dict['gt_name']) + "gt_vs_pred_cm.csv"))

    frame_by_frame_results = [total_gts, total_predictions, abs_perc_difference_w]

    print("Testing terminated.")

    return report_path, frame_by_frame_results


if __name__ == '__main__':
    # Parse input and start main
    parser = argparse.ArgumentParser(description='Testing: compute video classification metrics')
    parser.add_argument('-parFile', action='store', dest='parFile',
                        help='path to the parameter file', required=True)
    args = parser.parse_args()

    # Read params from input yml file
    if not os.path.exists(args.parFile):
        raise Exception("Parameter file %s does not exist" % args.parFile)
    print("Parsing parfile: %s " % args.parFile)
    with open(args.parFile, 'r') as stream:
        pars = yaml.safe_load(stream)

    _ = testing(pars)
