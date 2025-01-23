#!/usr/bin/env python

""" Run inference with network on a dataset.
If a testing_folder is specified, the code also runs testing.
This code is also used for unit test and TRT-pytorch testing.

Usage:
CUDA_VISIBLE_DEVICES=5 python3 inference.py -parFile real_colon/inference_v5.yml
"""
import numpy as np
import sys
import argparse
import yaml
import shutil
import torch
import os
import pickle

from src.data_loader import embeddings_dataset
from src.models import mstcn, asformer

def main(args):
    # Read params from input yml file
    if not os.path.exists(args.par_file):
        raise Exception("Parameter file %s does not exist" % (args.par_file))
    print("Parsing parfile: %s " % (args.par_file))
    with open(args.par_file, 'r') as stream:
        pars = yaml.safe_load(stream)
    print("Started inference with the following input pars: \n", pars)
    pars["general"]["output_folder"] = core.resolvePathMacros(pars["general"]["output_folder"])
    os.makedirs(pars["general"]["output_folder"], exist_ok=True)

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

    model = mstcn_utils.reload_tcn_from_model_state_dict(model, pars["model"]["model_path"])

    # load model on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create inference real_colon loader
    inference_dataset = embeddings_dataset.EmbeddingsDataset(pars["data_loader"]["valid"],
                                                             phase="validation",
                                                             prepare_dataset=pars["data_loader"]["prepare_dataset"],
                                                             temp_folder=core.resolvePathMacros(
                                                               pars["data_loader"]["temp_folder"]),
                                                             n_of_outputs=len(pars["model"]["n_classes"]))
    inference_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=1,
        num_workers=1,
        collate_fn=embeddings_dataset.custom_collate_batch,
        shuffle=False,
        pin_memory=False)

    # Run inference
    model.eval()
    with torch.no_grad():
        for batch_idx, (matrix, _, video_name, list_of_frame_names, _) in enumerate(inference_loader):
            net_input = matrix.to(device)

            # video embeddings needs to have dimension (N, C, L) in order to be passed into CNN
            net_input = net_input.transpose(1, 2)
            model_output = model(net_input)
            model_output = model_output[-1]
            video_name = video_name[0]
            list_of_frame_names = list_of_frame_names[0]

            if pars["general"]["check_with_trt_output"] != "":
                # Check the result with a TRT output
                path = core.resolvePathMacros(pars["general"]["check_with_trt_output"])

                # Extract the base name of the video.pkl containing the precomputed results (i.e., the file name)
                file_name = os.path.basename(path).replace('.pkl', '')
                required_part = file_name.split('_')[-2] + '_' + file_name.split('_')[-1]
                if required_part == video_name.split('_')[-2] + '_' + video_name.split('_')[-1]:
                    with open(path, 'rb') as f:
                        trt_out_data = pickle.load(f)
                    print("Video: ", video_name)
                    print("TRT output: ", trt_out_data[0])
                    print("Recomputed output: ", model_output.detach().cpu().numpy())

                    # Check for discrepancies
                    np.testing.assert_allclose(model_output.cpu().numpy().flatten(),
                                               trt_out_data[0],
                                               rtol=10 - 4)

            elif pars["general"]["check_with_previous_output"] != "":
                # Check the result with a previously computed pytorch output
                path = core.resolvePathMacros(pars["general"]["check_with_previous_output"])

                # Extract the base name of the video.pkl containing the precomputed results (i.e., the file name)
                file_name = os.path.basename(path).replace('.pkl', '')
                required_part = file_name.split('_')[-2] + '_' + file_name.split('_')[-1]
                if required_part == video_name.split('_')[-2] + '_' + video_name.split('_')[-1]:

                    # Load previous output for the same video
                    with open(path, 'rb') as f:
                        prev_data = pickle.load(f)
                    print("Video: ", video_name)
                    print("Previous output: ", prev_data['model_output'])
                    print("New inference output: ", model_output.detach().cpu().numpy())

                    np.testing.assert_allclose(model_output.detach().cpu().numpy(),
                                               prev_data['model_output'][0],
                                               atol=10 - 2)

            else:
                print("Running inference on video: ", video_name)

                # Save the output prediction for the current video in the output folder
                op = os.path.join(pars["general"]["output_folder"], video_name + ".pkl")
                with open(op, "wb") as fp:
                    if isinstance(model_output, tuple):
                        pickle.dump({"model_output": [model_output[0].detach().cpu().numpy(),
                                                      model_output[1].detach().cpu().numpy()],
                                     "list_of_frame_names": list_of_frame_names}, fp)
                    else:
                        pickle.dump({"model_output": [model_output.detach().cpu().numpy()],
                                     "list_of_frame_names": list_of_frame_names}, fp)

    print("Inference terminated.")

    # run testing on inference output if a testing folder is specified
    if pars["general"]["testing_folder"] != "":
        pars["general"]["inference_output"] = core.resolvePathMacros(pars["general"]["output_folder"])
        pars["general"]["output_folder"] = core.resolvePathMacros(pars["general"]["testing_folder"])
        _ = testing.testing(pars)


if __name__ == '__main__':
    # Parse input and start main
    parser = argparse.ArgumentParser(description='Inference with a TCN model')
    parser.add_argument('-parFile', action='store', dest='par_file',
                        help='path to the parameter file', required=True)
    args = parser.parse_args()
    main(args)
