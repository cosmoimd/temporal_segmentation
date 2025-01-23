#!/usr/bin/env python

"""
Visualize predictions of temporal segmentation models. This code was used for a technical paper.

Usage:
    python3 models_pred_visualisation.py -parFile real_colon/models_pred_visualisation.yml
"""

import numpy as np
import sys
import argparse
import yaml
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import pandas as pd


def aggregate(pars):
    """
    Compute classification testing metrics on TCN inference output.

    Args:
        pars (dict): A dictionary containing parameters loaded from the YAML file.
    """
    print("Parameters:\n", pars)

    f1s = []
    wf1s = []
    w_average_jaccard_per_class = []
    average_withdrawal = []
    average_jaccard_per_class = []

    for csv_test_result_path in pars["folds_testing_csvs"]:
        csv_test_result = pd.read_csv(core.resolvePathMacros(csv_test_result_path))
        f1s.append(csv_test_result['f1-score'])
        wf1s.append(csv_test_result['f1-score'].to_list()[-1])
        average_jaccard_per_class.append(csv_test_result['jaccard_per_class'].to_list()[-1])
        average_withdrawal.append(csv_test_result['abs_perc_difference_w'].to_list()[-1])
        w_average_jaccard_per_class.append(csv_test_result['weighted_jaccard_per_class'].to_list()[-1])

    all_f1_scores = pd.concat(f1s, axis=1)
    all_f1_scores.index = csv_test_result["Unnamed: 0"]
    all_f1_scores.columns = ['f1_fold_1', 'f1_fold_2', 'f1_fold_3', 'f1_fold_4', 'f1_fold_5']
    all_f1_scores['mean_f1'] = all_f1_scores.mean(axis=1)
    all_f1_scores['std_f1'] = all_f1_scores.std(axis=1)
    all_f1_scores.loc['jaccard'] = average_jaccard_per_class + [np.mean(average_jaccard_per_class),
                                                                np.std(average_jaccard_per_class)]
    all_f1_scores.loc['weighted_jaccard'] = w_average_jaccard_per_class + [np.mean(w_average_jaccard_per_class),
                                                                           np.std(w_average_jaccard_per_class)]
    all_f1_scores.loc['average_withdrawal'] = average_withdrawal + [np.mean(average_withdrawal),
                                                                    np.std(average_withdrawal)]

    print(all_f1_scores)
    all_f1_scores.to_csv(core.resolvePathMacros(pars["output_csv"]))

    print("Aggregation terminated.")
    return all_f1_scores['mean_f1'][-1], wf1s, average_jaccard_per_class, average_withdrawal


def plot_predictions(ax, predictions, title, class_colors):
    """
    Plot predictions for a given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the predictions.
        predictions (list): List of predictions.
        title (str): Title for the plot.
        class_colors (dict): Dictionary mapping class indices to colors.
    """
    ax.set_xlim(0, len(predictions))
    for i, pred in enumerate(predictions):
        ax.axvline(x=i, color=class_colors[pred], linewidth=4)  # Increased linewidth for better visibility

    ax.text(-0.02, 0.5, title, va='center', ha='right', transform=ax.transAxes)
    ax.set_yticks([])
    ax.set_xticks([])


def main(args):
    """
    Main function to read parameters and visualize model predictions.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    # Read parameters from input YAML file
    if not os.path.exists(args.par_file):
        raise FileNotFoundError(f"Parameter file {args.par_file} does not exist")
    print(f"Parsing parameter file: {args.par_file}")
    with open(args.par_file, 'r') as stream:
        pars = yaml.safe_load(stream)

    print(f"Started inference with the following input parameters:\n{pars}")
    pars["general"]["output_folder"] = core.resolvePathMacros(pars["general"]["output_folder"])
    os.makedirs(pars["general"]["output_folder"], exist_ok=True)

    # Save current git commit and config parameters in the output folder
    git.log_git_revision_to_file(pars["general"]["output_folder"])
    shutil.copy2(args.par_file, pars["general"]["output_folder"])

    video_info_csv = pd.read_csv(core.resolvePathMacros(pars["general"]["video_info_csv"]))

    # Define class colors
    class_colors = {
        1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd',
        6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f', 9: '#bcbd22', 10: '#17becf', 1000: '#17becf'
    }

    class_names = ['Outside', 'Insertion', 'Ceacum', 'Ileum', 'Ascending', 'Transverse', 'Descending', 'Sigmoid',
                   'Rectum', 'Uncertain']
    patches = [mpatches.Patch(color=color, label=name) for name, color in zip(class_names, class_colors.values())]

    ymls_in_the_cfg_file = core.find_key_in_dict(pars, ".yml")
    for path_to_yml in ymls_in_the_cfg_file:
        shutil.copy2(os.path.join(current_folder, path_to_yml), pars["general"]["output_folder"])

    dataset_data = pars["data_loader"]["valid"]

    # Loop over folds and datasets to print results on videoa of different models
    i_gt = 0
    for fold, datasets_fold in enumerate(dataset_data):
        for i_dataset, dataset_fold in enumerate(datasets_fold):
            with open(core.resolvePathMacros(dataset_fold["video_list"]), "r") as f:
                video_names = [line.rstrip() for line in f]

            for video_name in video_names:
                cvn = video_name[:-4] + ".pkl" if ".csv" in video_name else video_name + ".pkl"
                csv_path = os.path.join(core.resolvePathMacros(dataset_fold["csv_path"][i_gt]), cvn[:-4] + ".csv")
                csv_file = pd.read_csv(csv_path)

                list_of_models_preds = []
                for i, model_testing_path in enumerate(pars["model"]["models_inference_results"]):
                    op = core.resolvePathMacros(os.path.join(model_testing_path, f"fold_{fold + 1}", cvn))
                    with open(op, "rb") as fp:
                        inference_output = pickle.load(fp)

                    list_of_frame_names = inference_output["list_of_frame_names"]
                    if i == 0:
                        list_of_frame_names = csv_file[csv_file["image_name"].isin(list_of_frame_names)][
                            "image_name"].to_list()
                        gt_name = dataset_fold["gt_name"][i_gt]
                        gt = csv_file[csv_file["image_name"].isin(list_of_frame_names)][gt_name].to_list()
                        gt = [dataset_fold["str_to_idx"][i_gt][g] for g in gt]
                        gt = [g + 1 for i, g in enumerate(gt) if g != 999]

                    model_output = inference_output["model_output"][i_gt]
                    pred = list(np.argmax(model_output, axis=2).flatten())
                    pred = [p for i, p in enumerate(pred) if gt[i] != 999]

                    list_of_models_preds.append([p + 1 for i, p in enumerate(pred) if gt[i] != 999])

                # Plotting
                fig, axes = plt.subplots(len(pars["model"]["model_names"]) + 1, 1, figsize=(10, 6), sharex=True)
                fig.subplots_adjust(hspace=0, wspace=0)
                plot_predictions(axes[0], gt, 'Ground Truth', class_colors)
                for i, pred in enumerate(list_of_models_preds):
                    plot_predictions(axes[i + 1], pred, pars["model"]["model_names"][i], class_colors)

                plt.tight_layout(rect=[0, 0, 1, 0.95])

                legend_bottom = -0.25 - (0.05 * len(pars["model"]["model_names"]))
                plt.legend(handles=patches, bbox_to_anchor=(0.5, legend_bottom), loc='upper center', ncol=len(patches),
                           frameon=False)

                plt.savefig(os.path.join(pars["general"]["output_folder"], f"fold_{fold + 1}_{cvn[:-4]}.pdf"),
                            format='pdf', dpi=300, bbox_inches="tight")

    # Loop over folds and datasets
    gts_res = []
    i_gt = 0
    for fold, datasets_fold in enumerate(dataset_data):
        for i_dataset, dataset_fold in enumerate(datasets_fold):
            with open(core.resolvePathMacros(dataset_fold["video_list"]), "r") as f:
                video_names = [line.rstrip() for line in f]

            for video_name in video_names:
                cvn = video_name[:-4] + ".pkl" if ".csv" in video_name else video_name + ".pkl"
                csv_path = os.path.join(core.resolvePathMacros(dataset_fold["csv_path"][i_gt]), cvn[:-4] + ".csv")
                csv_file = pd.read_csv(csv_path)

                for i, model_testing_path in enumerate(pars["model"]["models_inference_results"]):
                    op = core.resolvePathMacros(os.path.join(model_testing_path, f"fold_{fold + 1}", cvn))
                    with open(op, "rb") as fp:
                        inference_output = pickle.load(fp)

                    list_of_frame_names = inference_output["list_of_frame_names"]
                    if i == 0:
                        list_of_frame_names = csv_file[csv_file["image_name"].isin(list_of_frame_names)][
                            "image_name"].to_list()
                        gt_name = dataset_fold["gt_name"][i_gt]
                        gt = csv_file[csv_file["image_name"].isin(list_of_frame_names)][gt_name].to_list()
                        gt = [dataset_fold["str_to_idx"][i_gt][g] for g in gt]
                        gts_res.append({"video_name": video_name,
                                        "gt": [g + 1 for i, g in enumerate(gt)]
                                        }
                                       )
    # List of datasets as they are named in the dataset release for my paper experiments
    dataset_name_list = ["set1_CB-17-08_split_procedures_WL_first_v20230206",
                         "set2_change_study",
                         "set3_102_maieron_stpolten",
                         "set4_sano-hospital"]
    counter = 15
    rc_videonames = [[], [], [], []]
    study_videos = [[], [], [], []]
    for row in video_info_csv.iterrows():
        # This is to find the dataset name
        if counter % 15 == 0:
            dataset_name = dataset_name_list[int(counter / 15) - 1]
        video_name = row[1]["video_name"]
        print("Working on video name: ", video_name)

        index_value = video_info_csv[video_info_csv["video_name"] == video_name]["index"].values[0]
        video_name_in_real_colon = f"{int(int(counter / 15)):03d}-{int(index_value) + 1:03d}"
        print("Which has in REAL-Colon the video name: ", video_name_in_real_colon)

        for el in gts_res:
            if el["video_name"] == "action_info_" + video_name:
                rc_videonames[int(counter / 15) - 1].append(video_name_in_real_colon)
                study_videos[int(counter / 15) - 1].append(el["gt"])

        counter += 1

    # Iterate over each dataset and plot GT for eac video a a gts folder
    for isv, sv in enumerate(study_videos):
        print("Printing dataset", isv + 1)

        for i, g in enumerate(sv):
            # Create a new figure and axes
            fig, ax = plt.subplots(figsize=(10, 6))
            # fig.subplots_adjust(hspace=0, wspace=0)

            predictions = g[::5]  # Sample every 5th prediction for plotting
            ax.set_xlim(0, len(predictions))

            # Plot vertical lines for predictions
            for ip, pred in enumerate(predictions):
                ax.axvline(x=ip, color=class_colors.get(pred, 'black'), linewidth=4)

            # Annotate with video name
            ax.text(-0.02, 0.5, rc_videonames[isv][i], va='center', ha='right', transform=ax.transAxes)
            ax.set_yticks([])
            ax.set_xticks([])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            legend_bottom = -0.25 - (0.05 * len(predictions))
            plt.legend(handles=patches, bbox_to_anchor=(0.5, legend_bottom), loc='upper center', ncol=len(patches),
                       frameon=False)

            # Save the figure
            output_file = os.path.join(pars["general"]["output_folder"], "gts", f"{rc_videonames[isv][i]}.png")
            plt.savefig(output_file, format='png', dpi=100, bbox_inches="tight")

            # Close the figure to free up memory
            plt.close(fig)

            print("Done")

    print("Code end.")


if __name__ == '__main__':
    # Parse input and start main
    parser = argparse.ArgumentParser(description='Visualize predictions by TCN models')
    parser.add_argument('-parFile', action='store', dest='par_file', help='Path to the parameter file', required=True)
    args = parser.parse_args()
    main(args)
