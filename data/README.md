# REAL-Colon Benchmark Setup Guide

## 1. REAL-Colon Dataset Download

The REAL-Colon dataset is available at [this link](https://doi.org/10.25452/figshare.plus.22202866), and the frame-wise annotations released with this work can be accessed [here](https://doi.org/10.6084/m9.figshare.26472913). Code to automatically download and process the dataset is available at [GitHub repository](https://github.com/cosmoimd/real-colon-dataset).

Please download the dataset at `./data/dataset/RC_dataset`.

## 2. Download Temporal Segmentation Annotations

Download the zip file containing all the CSV files for annotations from [this link](https://doi.org/10.6084/m9.figshare.26472913) and unzip it at `./data/dataset/RC_annotation`.

## 3. Run Feature Extraction

Feature extraction script encodes video frames into their latent representations using a predefined encoder model (ResNet50 pretrained on ImageNet). It supports augmentation and handles multiple videos in batches.

Usage:
```bash
CUDA_VISIBLE_DEVICES=0 python3 ./data/feature_extraction/feature_extraction.py --config ./data/feature_extraction/ymls/feature_extraction_1x_RC.yml
CUDA_VISIBLE_DEVICES=0 python3 ./data/feature_extraction/feature_extraction.py --config ./data/feature_extraction/ymls/feature_extraction_5x_aug_RC.yml
```

## 4. Train/Validate/Test Split for 4-fold and 5-fold Experiments
These splits have been saved at `./data/dataset/RC_lists` under the `4_fold` and `5_fold` directories.

## 5. Create Embeddings Dataset
After ensuring that feature extraction was successful, this script checks and creates a dataset for the classification TCN application. It saves a pickle file for every video in the dataset. Each pickle file contains a dictionary where `"video_embeddings"` is a numpy array of the embedded video features, which can be shaped `[1, temporal_size, latent_size]` or `[n_augmentations, temporal_size, latent_size]`; and a list of frame image names at key `"image_names"`.

Usage:
```bash
python3 create_embeddings_datasets.py --config data/emb_datasets_v2_mbmmx.yml
```

## Examples of Image and Annotations in the dataset
<img src="./images/frame_variability_visualization.png" alt="Detailed Temporal Segmentation Visualization" width="50%">

## References
Biffi, C., Antonelli, G., Bernhofer, S., Hassan, C., Hirata, D., Iwatate, M., Maieron, A., Salvagnini, P., & Cherubini, A. (2024). REAL-Colon: A dataset for developing real-world AI applications in colonoscopy. Scientific Data, 11(1), 539. DOI:10.1038/s41597-024-03359-0