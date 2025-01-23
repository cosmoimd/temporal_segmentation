#!/usr/bin/env python

"""Run inference with a network on a dataset in k-fold setting.
If a testing_folder is specified, the code also runs testing.

Usage:
CUDA_VISIBLE_DEVICES=4 python3 inference_testing_on_folds.py -parFile real_colon/user_configs/real_colon/.yml &
"""

import os
import sys
import argparse
import yaml
import shutil
import glob
import csv
import pickle
import numpy as np
import pandas as pd
import torch

from src.data_loader import embeddings_dataset
from src.models import mstcn, asformer

def aggregate(params):
    """
    Computes classification testing metrics on TCN inference output.

    Args:
        params (dict): A parameters dictionary
    """
    f1s = []
    wf1s = []
    avg_jaccard_per_class = []
    avg_withdrawal = []
    weighted_jaccard_per_class = []

    for csv_test_result_path in params["folds_testing_csvs"]:
        csv_test_result = pd.read_csv(core.resolvePathMacros(csv_test_result_path))
        f1s.append(csv_test_result['f1-score'])
        wf1s.append(csv_test_result['f1-score'].to_list()[-1])
        avg_jaccard_per_class.append(csv_test_result['jaccard_per_class'].to_list()[-1])
        avg_withdrawal.append(csv_test_result['abs_perc_difference_w'].to_list()[-1])
        weighted_jaccard_per_class.append(csv_test_result['weighted_jaccard_per_class'].to_list()[-1])

    all_f1_scores = pd.concat(f1s, axis=1)
    all_f1_scores.index = csv_test_result["Unnamed: 0"]
    all_f1_scores.columns = params["general"]["folds_values"]
    all_f1_scores['mean_f1'] = all_f1_scores.mean(axis=1)
    all_f1_scores['std_f1'] = all_f1_scores.std(axis=1)
    all_f1_scores.loc['jaccard'] = avg_jaccard_per_class + [np.mean(avg_jaccard_per_class),
                                                            np.std(avg_jaccard_per_class)]
    all_f1_scores.loc['weighted_jaccard'] = weighted_jaccard_per_class + [np.mean(weighted_jaccard_per_class),
                                                                          np.std(weighted_jaccard_per_class)]
    all_f1_scores.loc['average_withdrawal'] = avg_withdrawal + [np.mean(avg_withdrawal),
                                                                np.std(avg_withdrawal)]
    print(all_f1_scores)
    all_f1_scores.to_csv(core.resolvePathMacros(params["output_csv"]))

    print("Aggregation terminated.")
    return all_f1_scores['mean_f1'][-1], wf1s, avg_jaccard_per_class, avg_withdrawal


def main(args):
    if not os.path.exists(args.par_file):
        raise FileNotFoundError(f"Parameter file {args.par_file} does not exist")

    with open(args.par_file, 'r') as stream:
        params = yaml.safe_load(stream)

    print("Started inference with the following input parameters: \n", params)

    params["general"]["output_folder"] = core.resolvePathMacros(params["general"]["output_folder"])
    os.makedirs(params["general"]["output_folder"], exist_ok=True)

    # Save current git commit and config params in output folder
    git.log_git_revision_to_file(params["general"]["output_folder"])
    shutil.copy2(args.par_file, params["general"]["output_folder"])

    # Save YAML files used in the config file
    ymls_in_the_cfg_file = core.find_key_in_dict(params, ".yml")
    for path_to_yml in ymls_in_the_cfg_file:
        shutil.copy2(os.path.join(current_folder, path_to_yml), params["general"]["output_folder"])

    # Define or reload model from a previous checkpoint
    if params["model"]["model_type"] in ["tcn", "mstcn", "cmstcn"]:
        model = mstcn.MS_TCN(**params["model"])
    elif params["model"]["model_type"] == "asformer":
        model = asformer.ASFormer(**params["model"])
    else:
        raise ValueError("Wrong model_type!")

    # Validation
    start, end, step = params["general"]["checkpoint_start"], params["general"]["checkpoint_end"], params["general"][
        "checkpoint_step"]
    file_names = [f"*epoch-{i}_*.pth" for i in range(start, end + 1, step)]
    checkpoints_results = []
    dataset_data = params["data_loader"]["valid"]
    output_folder_root = params["general"]["output_folder"]
    best_csvs_per_fold = [""] * len(params["general"]["folds_values"])
    best_checkpoint_per_fold = [""] * len(params["general"]["folds_values"])
    best_values_per_fold = [0.] * len(params["general"]["folds_values"])

    # Loop over checkpoints
    for icheck, checkpoint in enumerate(file_names):
        current_folder_checkpoint = os.path.join(output_folder_root, checkpoint)
        os.makedirs(current_folder_checkpoint, exist_ok=True)
        params["folds_testing_csvs"] = []

        # Loop over dataset folds
        for fold, dataset_fold in enumerate(dataset_data):
            fold_folder = os.path.join(current_folder_checkpoint, "fold_" + str(fold + 1))
            os.makedirs(fold_folder, exist_ok=True)

            # Load model on device
            model = mstcn_utils.reload_tcn_from_model_state_dict(model,
                                                                 glob.glob(os.path.join(core.resolvePathMacros(
                                                                     params["model"]['model_path'][fold]),
                                                                                        checkpoint))[0])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Create inference real_colon loader
            inference_dataset = embeddings_dataset.EmbeddingsDataset(dataset_fold,
                                                                     phase="validation",
                                                                     prepare_dataset=params["data_loader"][
                                                                         "prepare_dataset"],
                                                                     temp_folder=core.resolvePathMacros(
                                                                         params["data_loader"]["temp_folder_valid"][
                                                                             fold]),
                                                                     n_of_outputs=len(params["model"]["n_classes"]))
            inference_loader = torch.utils.data.DataLoader(
                inference_dataset,
                batch_size=1,
                num_workers=1,
                collate_fn=embeddings_dataset.custom_collate_batch,
                shuffle=False,
                pin_memory=True
            )

            # Run inference
            model.eval()
            with torch.no_grad():
                for matrix, _, video_name, list_of_frame_names, _ in inference_loader:
                    net_input = matrix.to(device).transpose(1, 2)
                    model_output = model(net_input)[-1]
                    video_name = video_name[0]
                    list_of_frame_names = list_of_frame_names[0]

                    op = os.path.join(fold_folder, video_name + ".pkl")
                    with open(op, "wb") as fp:
                        pickle.dump({"model_output": [model_output.detach().cpu().numpy()],
                                     "list_of_frame_names": list_of_frame_names}, fp)

            # Run testing on inference output for this fold
            params["general"]["inference_output"] = fold_folder
            params["general"]["output_folder"] = fold_folder
            params["data_loader"]["valid"] = dataset_fold
            folds_testing_csvs, frame_by_frame_results = testing.testing(params)
            params["folds_testing_csvs"].append(folds_testing_csvs)

        params["output_csv"] = os.path.join(current_folder_checkpoint, "result.csv")
        mean_weighted_f1, wf1s, avg_jaccard_per_class, avg_withdrawal = aggregate(params)
        checkpoints_results.append({
            "checkpoint": checkpoint,
            "mean_weighted_f1": mean_weighted_f1,
            "wf1s": wf1s,
            "average_jaccard_per_class": avg_jaccard_per_class,
            "average_withdrawal": avg_withdrawal,
        })

        if icheck > params["general"]["min_check"]:
            for isplit, metric in enumerate(wf1s):
                if metric >= best_values_per_fold[isplit]:
                    best_values_per_fold[isplit] = metric
                    best_csvs_per_fold[isplit] = params["folds_testing_csvs"][isplit]
                    best_checkpoint_per_fold[isplit] = checkpoint

    # Save checkpoints results in a big table
    num_folds = len(checkpoints_results[0]['wf1s'])
    column_names = ['checkpoint', 'mean_weighted_f1'] + [f'fold_{i + 1}_wf1' for i in range(num_folds)]
    rows = []
    for result in checkpoints_results:
        row = [result['checkpoint'], result['mean_weighted_f1']] + result['wf1s']
        rows.append(row)
    with open(os.path.join(output_folder_root, "checkpoints_results.csv"), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(rows)

    # Testing
    params["folds_testing_csvs"] = []
    testing_folder = os.path.join(output_folder_root, "TESTING")
    os.makedirs(testing_folder, exist_ok=True)
    frame_by_frame_results_list = []
    for ifold, checkpoint in enumerate(best_checkpoint_per_fold):
        fold_folder = os.path.join(testing_folder, "fold_" + str(ifold + 1))
        os.makedirs(fold_folder, exist_ok=True)

        model = mstcn_utils.reload_tcn_from_model_state_dict(model,
                                                             glob.glob(os.path.join(core.resolvePathMacros(
                                                                 params["model"]['model_path'][ifold]),
                                                                                    checkpoint))[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inference_dataset = embeddings_dataset.EmbeddingsDataset(params["data_loader"]["test"][ifold],
                                                                 phase="validation",
                                                                 prepare_dataset=params["data_loader"][
                                                                     "prepare_dataset"],
                                                                 temp_folder=core.resolvePathMacros(
                                                                     params["data_loader"]["temp_folder_test"][ifold]),
                                                                 n_of_outputs=len(params["model"]["n_classes"]))
        inference_loader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=embeddings_dataset.custom_collate_batch,
            shuffle=False,
            pin_memory=True
        )

        model.eval()
        with torch.no_grad():
            for matrix, _, video_name, list_of_frame_names, _ in inference_loader:
                net_input = matrix.to(device).transpose(1, 2)
                model_output = model(net_input)[-1]
                video_name = video_name[0]
                list_of_frame_names = list_of_frame_names[0]

                op = os.path.join(fold_folder, video_name + ".pkl")
                with open(op, "wb") as fp:
                    pickle.dump({"model_output": [model_output.detach().cpu().numpy()],
                                 "list_of_frame_names": list_of_frame_names}, fp)

        params["general"]["inference_output"] = fold_folder
        params["general"]["output_folder"] = fold_folder
        params["data_loader"]["valid"] = params["data_loader"]["test"][ifold]
        folds_testing_csvs, frame_by_frame_results = testing.testing(params)
        params["folds_testing_csvs"].append(folds_testing_csvs)
        frame_by_frame_results_list.append(frame_by_frame_results)

    from sklearn.metrics import classification_report
    total_gts = []
    total_predictions = []
    abs_perc_difference_w = []
    for results in frame_by_frame_results_list:
        total_gts += results[0]
        total_predictions += results[1]
        abs_perc_difference_w += results[2]

    classification_rep_save = classification_report(total_gts, total_predictions, output_dict=True)
    df_report = pd.DataFrame(classification_rep_save).transpose()
    str_to_idx_items = params["data_loader"]["test"][0][0]["str_to_idx"][0].items()
    df_report["precision"] *= 100
    df_report["recall"] *= 100
    df_report["f1-score"] *= 100
    jaccard_per_class, avg_jaccard = testing.compute_jaccard_index(total_predictions, total_gts)
    weighted_jaccard_per_class = testing.compute_weighted_jaccard_index(total_predictions, total_gts)
    df_report["abs_perc_difference_w"] = [np.mean(abs_perc_difference_w) * 100] * df_report.shape[0]
    df_report["jaccard_per_class"] = jaccard_per_class + [avg_jaccard] * (df_report.shape[0] - 9)
    df_report["weighted_jaccard_per_class"] = weighted_jaccard_per_class + [weighted_jaccard_per_class] * (
            df_report.shape[0] - 9)
    df_report["jaccard_per_class"] *= 100
    df_report["weighted_jaccard_per_class"] *= 100
    df_report = np.round(df_report, decimals=1)
    df_report = df_report.rename(index=dict((str(v), k) for k, v in str_to_idx_items))
    df_report.to_csv(os.path.join(output_folder_root, "test_overall_gt_vs_pred_perFrame.csv"))

    params["output_csv"] = os.path.join(testing_folder, "result.csv")
    _, _, _, _ = aggregate(params)

    print("Code end.")


if __name__ == '__main__':
    # Parse input and start main
    parser = argparse.ArgumentParser(description='Inference with a TCN model for k-fold CV')
    parser.add_argument('-parFile', required=True, help='Path to the parameter file')
    args = parser.parse_args()
    main(args)
