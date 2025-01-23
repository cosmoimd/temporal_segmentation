# Temporal Segmentation of Full-Procedure Colonoscopy Videos

## Overview
This repository accompanies the paper **"ColonTCN: Temporal Segmentation of Colonoscopy Videos"**. It provides the implementation of ColonTCN, a Temporal Convolutional Network-based approach for segmenting colonoscopy videos into anatomical sections and procedural phases. The project leverages a benchmark dataset derived from the annotated REAL-Colon dataset, which features 2.7 million frames across 60 full-procedure videos.

![Detailed Temporal Segmentation Visualization](./data/images/visualisation.png)

This repository includes:

- **Annotated Dataset:** Access to the annotated REAL-Colon dataset, which includes frame-level labels for anatomical locations and colonoscopy phases.
- **ColonTCN Model:** Implementation of the ColonTCN architecture, designed to efficiently capture long-term temporal dependencies in full-procedure colonoscopy videos.
- **Evaluation Scripts for OP:** Scripts for evaluating model performance using the two proposed k-fold cross-validation protocol proposed in the paper.

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your_username/ColonTCN.git
```

### 2. Set up the environment
Install the necessary dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Dataset and Open-Access Benchmark
The annotated dataset used in this study is the REAL-colon dataset. [Click here](./data/README.md) for instructions to automatically download, extract the data, and load splits to benchmark the 

## Training

This script handles the training of models based on specified configurations and datasets.

### Usage
To start training, use the command:
CUDA_VISIBLE_DEVICES=0 python3 training.py --config training_config.yml

### Description
The training script initializes a model from a configuration file, trains it with specified datasets, and saves checkpoints and final model weights. It supports various loss functions and optimizations to enhance training efficiency and effectiveness.

## Inference

This script performs inference on provided datasets using a pretrained model and outputs the classification or detection results.

### Usage
To run the inference script:
CUDA_VISIBLE_DEVICES=1 python3 inference.py --config your_config_file.yml

### Description
The script loads a trained model specified in the configuration file and performs inference on the data specified. It outputs the results, which include classification labels or detected object coordinates, depending on the model used.

## Inference Testing on Folds

This script is designed to test the inference results across different folds of data, often used in cross-validation processes.

### Usage
To execute the script:
python3 inference_testing_on_folds.py --config fold_config.yml

### Description
The script loads inference results from different folds, evaluates them collectively, and computes aggregated statistics to analyze model performance across all folds. This is useful for understanding the model's consistency and effectiveness in cross-validated settings.
## Testing

This script is used to compute video classification frame-wise metrics, label files, and timeseries plots based on the inference output.

### Usage
To use the testing script:
python3 testing.py --config testing_config.yml

### Description
The script evaluates the performance of a model using ground truth data versus predicted data from the inference output. It provides detailed metrics such as Jaccard index, precision, recall, and F1-score, and generates plots to visualize performance over time.

## Profiling

This script profiles the model's performance, focusing on computational efficiency such as inference time and memory usage.

### Usage
Execute the script by running:
python3 profiling.py --config profiling_config.yml

### Description
It measures various performance metrics of the model during inference, including computation time and resource usage. This data is crucial for optimizing models for deployment in resource-constrained environments.
