#!/usr/bin/env python

"""Test shared ColonTCN models on the RC benchmark

Usage:
CUDA_VISIBLE_DEVICES=0 python3 src/test_shared_model.py -parFile ymls/inference/test_shared_4fold_colontcn.yml
CUDA_VISIBLE_DEVICES=0 python3 src/test_shared_model.py -parFile ymls/inference/test_shared_5fold_colontcn.yml
"""

import os
import sys
import argparse
import yaml
from sklearn.metrics import classification_report
import csv
import pickle
import numpy as np
import pandas as pd
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

from src.models.factory import ModelFactory
from src.data_loader.embeddings_dataset import EmbeddingsDataset, custom_collate_batch
from src.testing import testing, compute_jaccard_index, compute_weighted_jaccard_index

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
        csv_test_result = pd.read_csv(csv_test_result_path)
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
    all_f1_scores.to_csv(params["output_csv"])

    print("Aggregation terminated.")
    return all_f1_scores['mean_f1'][-1], wf1s, avg_jaccard_per_class, avg_withdrawal


def main(args):
    if not os.path.exists(args.parFile):
        raise FileNotFoundError(f"Parameter file {args.parFile} does not exist")
    with open(args.parFile, 'r') as stream:
        params = yaml.safe_load(stream)
    os.makedirs(params["general"]["output_folder"], exist_ok=True)

    # Create model using ModelFactory
    model_factory = ModelFactory(params["model"])
    model = model_factory.create_model(load_checkpoint=False)
    output_folder_root = params["general"]["output_folder"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing
    params["folds_testing_csvs"] = []
    os.makedirs(output_folder_root, exist_ok=True)
    frame_by_frame_results_list = []
    for ifold, checkpoint in enumerate(params["model"]["model_path"]):
        fold_folder = os.path.join(output_folder_root, "fold_" + str(ifold + 1))
        os.makedirs(fold_folder, exist_ok=True)

        print("Loading checkpoint: ", checkpoint)
        model = model_factory.load_checkpoint(model, checkpoint)

        inference_dataset = EmbeddingsDataset(params["data_loader"]["test"][ifold],
                                             phase="inference",
                                             prepare_dataset=params["data_loader"]["prepare_dataset"],
                                             rc_csv=params["data_loader"]["rc_csv"],
                                             temp_folder=params["data_loader"]["temp_folder_test"][ifold])
        inference_loader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=custom_collate_batch,
            shuffle=False,
            pin_memory=False
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
        folds_testing_csvs, frame_by_frame_results = testing(params)
        params["folds_testing_csvs"].append(folds_testing_csvs)
        frame_by_frame_results_list.append(frame_by_frame_results)

    total_gts = []
    total_predictions = []
    abs_perc_difference_w = []
    for results in frame_by_frame_results_list:
        total_gts += results[0]
        total_predictions += results[1]
        abs_perc_difference_w += results[2]

    classification_rep_save = classification_report(total_gts, total_predictions, output_dict=True)
    df_report = pd.DataFrame(classification_rep_save).transpose()
    str_to_idx_items = params["data_loader"]["test"][0][0]["str_to_idx"].items()
    df_report["precision"] *= 100
    df_report["recall"] *= 100
    df_report["f1-score"] *= 100
    jaccard_per_class, avg_jaccard = compute_jaccard_index(total_predictions, total_gts)
    weighted_jaccard_per_class = compute_weighted_jaccard_index(total_predictions, total_gts)
    df_report["abs_perc_difference_w"] = [np.mean(abs_perc_difference_w) * 100] * df_report.shape[0]
    df_report["jaccard_per_class"] = jaccard_per_class + [avg_jaccard] * (df_report.shape[0] - 9)
    df_report["weighted_jaccard_per_class"] = jaccard_per_class + [weighted_jaccard_per_class] * (df_report.shape[0] - 9)
    df_report["jaccard_per_class"] *= 100
    df_report["weighted_jaccard_per_class"] *= 100
    df_report = np.round(df_report, decimals=1)
    df_report = df_report.rename(index=dict((str(v), k) for k, v in str_to_idx_items))
    df_report.to_csv(os.path.join(output_folder_root, "test_overall_gt_vs_pred_perFrame.csv"))

    params["output_csv"] = os.path.join(params["general"]["output_folder"], "result.csv")
    _, _, _, _ = aggregate(params)

    print("Code end.")


if __name__ == '__main__':
    # Parse input and start main
    parser = argparse.ArgumentParser(description='Inference with a TCN model for k-fold CV')
    parser.add_argument('-parFile', required=True, help='Path to the parameter file')
    args = parser.parse_args()
    main(args)
