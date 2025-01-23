#!/usr/bin/env python

""" Compute video classification frame-wise metrics, lab files and timeseries plots on an inference output.

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
    pars["general"]["output_folder"] = core.resolvePathMacros(pars["general"]["output_folder"])
    os.makedirs(pars["general"]["output_folder"], exist_ok=True)
    compute_timeseries = pars["general"]["compute_timeseries"]
    compute_labs = pars["general"]["compute_labs"]
    save_wrong_frames = pars["general"]["save_wrong_frames"]
    exclude_borders = pars["general"]["exclude_borders"]

    # GTs and predictions to compute metrics aggregated over all datasets
    if len(pars["data_loader"]["valid"][0]['gt_name']) == 2:
        total_gts_0 = []  # overall list of all the frame gts from all the videos
        total_predictions_0 = []  # overall list of all the predictions from all the videos
        total_gts_1 = []  # overall list of all the frame gts from all the videos
        total_predictions_1 = []  # overall list of all the predictions from all the videos
    else:
        total_gts = []  # overall list of all the frame gts from all the videos
        total_predictions = []  # overall list of all the predictions from all the videos

    # loop over datasets, one output folder for each datasets
    for dataset_dict in pars["data_loader"]["valid"]:
        # create dataset output folder and labs subfolder
        current_output_folder = os.path.join(pars["general"]["output_folder"], dataset_dict["dataset_name"])
        os.makedirs(current_output_folder, exist_ok=True)

        # List of video filenames to study
        with open(core.resolvePathMacros(dataset_dict["video_list"]), "r") as f:
            video_names = [line.rstrip() for line in f]

        # Loop over GTs, given a GT loop over videos in that dataset
        abs_perc_difference_w = []
        for i_gt, gt_name in enumerate(dataset_dict['gt_name']):
            current_output_folder = os.path.join(pars["general"]["output_folder"],
                                                 dataset_dict["dataset_name"], gt_name)
            os.makedirs(current_output_folder, exist_ok=True)

            # Run testing, video by video
            # Collect GTs and predictions to compute dataset metrics
            tot_correct = 0
            tot_counter = 0
            gts = []  # overall list of all the frame gts from all the dataset videos
            predictions = []  # overall list of all the predictions from all the dataset videos

            for video_name in video_names:
                if ".csv" in video_name:
                    cvn = video_name[:-4] + ".pkl"
                else:
                    cvn = video_name + ".pkl"

                if "action_info" in cvn:
                    only_cvn = cvn[12:]

                # Load model output prediction for video_name
                op = core.resolvePathMacros(os.path.join(pars["general"]["inference_output"], cvn))
                with open(op, "rb") as fp:
                    inference_output = pickle.load(fp)
                print("Computing testing metrics and plots for video: ", video_name)

                # Retrieve computed time series predictions
                model_output = inference_output["model_output"][i_gt]
                pred = list(np.argmax(model_output, axis=2).flatten())
                list_of_frame_names = inference_output["list_of_frame_names"]

                # process dataset in csv (input of classification_rnn)
                csv_path = os.path.join(core.resolvePathMacros(dataset_dict["csv_path"][i_gt]), cvn[:-4] + ".csv")
                csv_file = pd.read_csv(csv_path)
                list_of_frame_names = csv_file[csv_file["image_name"].isin(list_of_frame_names)][
                    "image_name"].to_list()
                gt = csv_file[csv_file["image_name"].isin(list_of_frame_names)][gt_name].to_list()
                gt = [dataset_dict["str_to_idx"][i_gt][g] for g in gt]

                # Remove frames at class 999
                pred = [p for i, p in enumerate(pred) if gt[i] != 999]
                list_of_frame_names = [p for i, p in enumerate(list_of_frame_names) if gt[i] != 999]
                gt = [g for i, g in enumerate(gt) if g != 999]

                abs_perc_difference_w.append(abs_perc_difference(pred, gt))

                if save_wrong_frames:
                    os.makedirs(os.path.join(current_output_folder, "wrong_frames"), exist_ok=True)
                    list_of_full_paths = csv_file[csv_file["image_name"].isin(list_of_frame_names)][
                        "full_path"].to_list()
                    if len(list_of_full_paths) != len(gt):
                        import pdb
                        pdb.set_trace()
                    list_of_wrong_fn = [[fn, gt[i], pred[i]]
                                        for i, fn in enumerate(list_of_full_paths) if gt[i] != pred[i]]

                    df = pd.DataFrame(list_of_wrong_fn, columns=['frame_id', 'GT', 'pred'])
                    df.to_csv(os.path.join(current_output_folder, "wrong_frames", cvn[:-4] + ".csv"))

                # create two subplots, one containing the line plots of the gt and the predicted time series,
                # the second plot containing their difference
                if compute_timeseries:
                    os.makedirs(os.path.join(current_output_folder, "timeseries"), exist_ok=True)
                    pred_shifted = np.array(
                        [p + 0.05 for p in pred])  # for visualisation, shift a bit vertically the predictions
                    diff = np.array([gt[i] - p for i, p in enumerate(pred)])  # for gt-prediction timeseries plot
                    classes_names = [key for key, value in pars["general"]["timeseries_classes_names"].items()]
                    x = np.arange(0, len(gt))
                    plt.figure(figsize=(400, 50))
                    plt.subplot(2, 1, 1)
                    plt.plot(x, gt, color='green', marker='o', label="GT")
                    plt.plot(x, pred_shifted, color='blue', marker='o', label="pred")
                    plt.legend(loc="upper left", fontsize=82)
                    plt.title('Predicted and GT time series', fontsize=52)
                    plt.xlabel('frame number', fontsize=48)
                    plt.xticks(fontsize=48)
                    plt.margins(x=0)
                    plt.ylabel('location', fontsize=48)
                    plt.yticks(np.arange(0, len(classes_names), 1), classes_names, fontsize=48)
                    plt.tick_params(axis="x", which='both', direction="in", labelbottom=True, labeltop=True,
                                    labelsize=44)
                    plt.tick_params(axis="y", which='both', direction="in", labelbottom=True, labeltop=True,
                                    labelsize=44)
                    plt.subplot(2, 1, 2)
                    plt.title('GT - prediction  time series', fontsize=52)
                    plt.plot(x, diff, 'go')
                    plt.xlabel('frame number', fontsize=48)
                    plt.ylabel('prediction difference', fontsize=48)
                    plt.tick_params(axis="x", which='both', direction="in", labelbottom=True, labeltop=True,
                                    labelsize=44)
                    plt.tick_params(axis="y", which='both', direction="in", labelbottom=True, labeltop=True,
                                    labelsize=44)
                    plt.margins(x=0)

                    # save the plot in output folder, one for each video
                    plt.savefig(
                        os.path.join(current_output_folder, "timeseries", cvn + ".pdf"),
                        format="pdf", bbox_inches="tight", dpi=300)
                    plt.close()

                # transform two labc with predictions and with prediction+GT for this video
                # (we generate lab files only on the original video real_colon)
                if compute_labs:
                    current_output_folder_labs = os.path.join(current_output_folder, "labs")
                    os.makedirs(current_output_folder_labs, exist_ok=True)
                    dataset_gt_labs_path = core.resolvePathMacros(dataset_dict["labs_path"])
                    path_to_lab_gt = os.path.join(dataset_gt_labs_path, only_cvn + ".lab")
                    idx_to_str = {dataset_dict["str_to_idx"][i_gt][k]: k for k in dataset_dict["str_to_idx"][i_gt]}
                    idx_to_gt_str = {dataset_dict["str_to_idx"][i_gt][k]: "gt_" + k for k in
                                     dataset_dict["str_to_idx"][i_gt]}
                    if os.path.exists(path_to_lab_gt):
                        # Loading video GT lab for the video and delete GT annotations and re-save thr lab file with
                        # predictions only as annotation for this video
                        lfr = labfile_reader_writer.LabFileReader()
                        summary_lab_file_data = lfr.read(path_to_lab_gt)
                        summary_lab_file_data['content']["Annotations"]['real_colon'] = []
                        summary_lab_file_data['content']["Elements"]['real_colon'] = []
                        lab_saver_params_dict = {"format": "labc", "output_filename": only_cvn + ".labc"}
                        lab_saver = output_saver.CharacterizationSaver(lab_saver_params_dict)
                        lab_saver.lwr.set_from_dict(summary_lab_file_data)
                        pred_upsampled = np.repeat(pred, pars["general"]["upsampling_factor_for_labs"])
                        pred_switches = timeseries_filters.list_to_switches(pred_upsampled.tolist())
                        pred_annotations = timeseries_filters.switches_to_annotations(pred_switches,
                                                                                      idx_to_str,
                                                                                      start_only=True)
                        fps = round(summary_lab_file_data['content']['Main Header']['fps'])
                        timestamp = csv_file["frame_id"][0].split("_")
                        start_id = round((int(timestamp[-1]) / 1000) * fps)
                        if start_id > 0:
                            # this means that you are working with action, ie a video not starting from 0
                            for i_pa, pa in enumerate(pred_annotations):
                                pred_annotations[i_pa]["rawId"] = pa["rawId"] + start_id
                                pred_annotations[i_pa]["id"] = pa["id"] + start_id

                            # this is an action lab
                            summary_lab_file_data = lfr.read(path_to_lab_gt)
                            pred_annotations.insert(0, summary_lab_file_data['content']["Annotations"]['real_colon'][0])
                            pred_annotations.append(summary_lab_file_data['content']["Annotations"]['real_colon'][-1])

                        lab_saver.append_annotations(pred_annotations)
                        lab_saver.write(output_folder=current_output_folder_labs)

                        # Loading video GT lab for the video and delete GT annotations and re-save thr lab file with
                        # predictions AND GT annotations as annotation for this video
                        lfr = labfile_reader_writer.LabFileReader()
                        summary_lab_file_data = lfr.read(path_to_lab_gt)
                        summary_lab_file_data['content']["Annotations"]['real_colon'] = []
                        summary_lab_file_data['content']["Elements"]['real_colon'] = []
                        lab_saver_params_dict = {"format": "labc",
                                                 "output_filename": only_cvn + "_with_gt.labc"}
                        lab_saver = output_saver.CharacterizationSaver(lab_saver_params_dict)
                        lab_saver.lwr.set_from_dict(summary_lab_file_data)
                        gt_upsampled = np.repeat(gt, pars["general"]["upsampling_factor_for_labs"]).tolist()
                        gt_switches = timeseries_filters.list_to_switches(gt_upsampled)
                        gt_annotations = timeseries_filters.switches_to_annotations(gt_switches,
                                                                                    idx_to_gt_str,
                                                                                    start_only=True)
                        lab_saver.append_annotations(gt_annotations + pred_annotations)
                        lab_saver.write(output_folder=current_output_folder_labs)
                    else:
                        print("This path does not exists, going in debug mode: ", path_to_lab_gt)
                        import pdb
                        pdb.set_trace()

                # Exclude borders+-delta from results computation
                if exclude_borders:
                    gt = [gt for i, g in enumerate(gt) if g != 999]
                    pred = [pred[i] for i, g in enumerate(gt) if g != 999]

                    delta = 30
                    new_gt = []
                    new_pred = []
                    for i, gt in enumerate(gt):
                        low = max(0, i - delta)
                        high = min(len(gt) - 1, i + delta)
                        uno = all(ele == gt[i] for ele in gt[low:high])
                        if uno:
                            new_gt.append(gt[i])
                            new_pred.append(pred[i])
                    gt = new_gt
                    pred = new_pred

                # Update total lists
                gts += gt
                predictions += pred

                # Update num and den for total accuracy
                tot_correct += sum(x == y for x, y in zip(pred, gt))
                tot_counter += len(pred)

            # Classification metrics and confusion matrix for total gt vs pred time series
            classification_rep_save = classification_report(gts, predictions, output_dict=True)
            jaccard_per_class, avg_jaccard = compute_jaccard_index(predictions, gts,
                                                                        num_classes=pars["model"]["n_classes"][0])
            weighted_jaccard_per_class = compute_weighted_jaccard_index(predictions, gts,
                                                                        num_classes=pars["model"]["n_classes"][0])
            df_report = pd.DataFrame(classification_rep_save).transpose()
            df_report["jaccard_per_class"] = jaccard_per_class + [avg_jaccard] * 3
            df_report["weighted_jaccard_per_class"] = (jaccard_per_class + [weighted_jaccard_per_class] * 3)

            # String to index items
            str_to_idx_items = dataset_dict['str_to_idx'][i_gt].items()

            # Save classification report using 1 decimal places (88.1) and get class names from str_to_idx
            df_report["precision"] *= 100
            df_report["recall"] *= 100
            df_report["f1-score"] *= 100
            df_report["jaccard_per_class"] *= 100
            df_report["weighted_jaccard_per_class"] *= 100
            df_report = np.round(df_report, decimals=1)
            df_report = df_report.rename(index=dict((str(v), k) for k, v in str_to_idx_items))
            df_report.to_csv(os.path.join(current_output_folder, str(gt_name) + "gt_vs_pred_perFrame.csv"))

            # Save confusion matrix with correct namings
            cm_images = confusion_matrix(gts, predictions)
            df_report = pd.DataFrame(cm_images)
            df_report = df_report.rename(index=dict((v, k) for k, v in str_to_idx_items))
            df_report = df_report.rename(columns=dict((v, k) for k, v in str_to_idx_items))
            df_report.to_csv(os.path.join(current_output_folder, str(gt_name) + "gt_vs_pred_cm.csv"))

            if len(pars["data_loader"]["valid"][0]['gt_name']) == 2:
                if i_gt == 0:
                    total_gts_0 += gts
                    total_predictions_0 += predictions
                else:
                    total_gts_1 += gts
                    total_predictions_1 += predictions
            else:
                total_gts += gts
                total_predictions += predictions


        if len(pars["data_loader"]["valid"][0]['gt_name']) == 2:
            # Classification metrics and confusion matrix aggregated over all datasets
            classification_rep_save = classification_report(total_gts_0, total_predictions_0, output_dict=True)
            df_report = pd.DataFrame(classification_rep_save).transpose()
            str_to_idx_items = dataset_dict['str_to_idx'][0].items()
            df_report["precision"] *= 100
            df_report["recall"] *= 100
            df_report["f1-score"] *= 100
            df_report = np.round(df_report, decimals=1)
            df_report = df_report.rename(index=dict((str(v), k) for k, v in str_to_idx_items))
            df_report.to_csv(os.path.join(pars["general"]["output_folder"], "0_gt_vs_pred_perFrame.csv"))
            cm_images = confusion_matrix(total_gts_0, total_predictions_0)
            df_report = pd.DataFrame(cm_images)
            df_report = df_report.rename(index=dict((v, k) for k, v in str_to_idx_items))
            df_report = df_report.rename(columns=dict((v, k) for k, v in str_to_idx_items))
            df_report.to_csv(os.path.join(pars["general"]["output_folder"], "0_gt_vs_pred_cm.csv"))

            classification_rep_save = classification_report(total_gts_1, total_predictions_1, output_dict=True)
            df_report = pd.DataFrame(classification_rep_save).transpose()
            str_to_idx_items = dataset_dict['str_to_idx'][1].items()
            df_report["precision"] *= 100
            df_report["recall"] *= 100
            df_report["f1-score"] *= 100
            df_report = np.round(df_report, decimals=1)
            df_report = df_report.rename(index=dict((str(v), k) for k, v in str_to_idx_items))
            df_report.to_csv(os.path.join(pars["general"]["output_folder"], "1_gt_vs_pred_perFrame.csv"))
            cm_images = confusion_matrix(total_gts_1, total_predictions_1)
            df_report = pd.DataFrame(cm_images)
            df_report = df_report.rename(index=dict((v, k) for k, v in str_to_idx_items))
            df_report = df_report.rename(columns=dict((v, k) for k, v in str_to_idx_items))
            df_report.to_csv(os.path.join(pars["general"]["output_folder"], "1_gt_vs_pred_cm.csv"))
        else:
            # Classification metrics and confusion matrix aggregated over all datasets
            classification_rep_save = classification_report(total_gts, total_predictions, output_dict=True)
            df_report = pd.DataFrame(classification_rep_save).transpose()
            str_to_idx_items = dataset_dict['str_to_idx'][0].items()
            df_report["precision"] *= 100
            df_report["recall"] *= 100
            df_report["f1-score"] *= 100
            jaccard_per_class, avg_jaccard = compute_jaccard_index(total_predictions, total_gts, num_classes=pars["model"]["n_classes"][0])
            weighted_jaccard_per_class = compute_weighted_jaccard_index(total_predictions, total_gts, num_classes=pars["model"]["n_classes"][0])
            df_report["abs_perc_difference_w"] = [np.mean(abs_perc_difference_w) * 100] * df_report.shape[0]
            df_report["jaccard_per_class"] = jaccard_per_class + [avg_jaccard] * 3
            df_report["weighted_jaccard_per_class"] = jaccard_per_class + [weighted_jaccard_per_class] * 3
            df_report["jaccard_per_class"] *= 100
            df_report["weighted_jaccard_per_class"] *= 100
            df_report = np.round(df_report, decimals=1)
            df_report = df_report.rename(index=dict((str(v), k) for k, v in str_to_idx_items))
            report_path = os.path.join(pars["general"]["output_folder"], str(gt_name) + "gt_vs_pred_perFrame.csv")
            df_report.to_csv(report_path)
            cm_images = confusion_matrix(total_gts, total_predictions)
            df_report = pd.DataFrame(cm_images)
            df_report = df_report.rename(index=dict((v, k) for k, v in str_to_idx_items))
            df_report = df_report.rename(columns=dict((v, k) for k, v in str_to_idx_items))
            df_report.to_csv(os.path.join(pars["general"]["output_folder"], str(gt_name) + "gt_vs_pred_cm.csv"))

    if len(pars["data_loader"]["valid"][0]['gt_name']) == 2:
        frame_by_frame_results = [[total_gts_0, total_gts_1], [total_predictions_0, total_predictions_1],
                                  abs_perc_difference_w]
    else:
        frame_by_frame_results = [total_gts, total_predictions, abs_perc_difference_w]

    print("Testing terminated.")

    return report_path, frame_by_frame_results


if __name__ == '__main__':
    # Parse input and start main
    parser = argparse.ArgumentParser(description='Test video classification metrics and plots on a inference output')
    parser.add_argument('-parFile', action='store', dest='par_file',
                        help='path to the parameter file', required=True)
    args = parser.parse_args()

    # Read params from input yml file
    if not os.path.exists(args.par_file):
        raise Exception("Parameter file %s does not exist" % args.par_file)
    print("Parsing parfile: %s " % args.par_file)
    with open(args.par_file, 'r') as stream:
        pars = yaml.safe_load(stream)

    _ = testing(pars)
