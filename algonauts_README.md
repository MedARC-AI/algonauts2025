# Algonauts Feature Extraction with DeepJuice

This script extracts features from the Algonauts 2025 dataset using DeepJuice and uploads them to HuggingFace.

## Setup

1. Make sure you have DeepJuice installed
2. Install required dependencies:
   ```
   pip install numpy pandas torch torchvision pillow tqdm h5py requests datalad huggingface_hub kagglehub
   ```
3. For Kaggle dataset download, authenticate kagglehub:
   ```
   # Set up Kaggle API credentials
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   
   # Or using kagglehub
   kagglehub login
   ```

## Dataset

You can get the Algonauts 2025 dataset in two ways:

1. From Kaggle (automatic): The script will use kagglehub to download from https://www.kaggle.com/datasets/ckadirt/algonauts2025nsl
2. Using Datalad (with --use-datalad flag): `datalad clone https://github.com/courtois-neuromod/algonauts_2025.competitors`

## Usage

### Basic Usage

```
python algonauts_feature_extraction.py --data-dir /path/to/algonauts_data --model-name vit_large_patch16_224
```

### Options

- `--data-dir`: Directory where the Algonauts dataset is stored
- `--output-dir`: Directory to save extracted features (default: 'extracted_features')
- `--model-name`: Name of the model to use for feature extraction (default: 'vit_large_patch16_224')
- `--models`: Multiple models to process (space-separated list, overrides --model-name)
- `--batch-size`: Batch size for processing (default: 32)
- `--frame-sampling-rate`: Sample 1 out of every N frames (default: 1 = use all frames)
- `--sampling-rates`: Multiple sampling rates to try (space-separated list, overrides --frame-sampling-rate)
- `--use-datalad`: Use Datalad to download the dataset
- `--upload-hf`: Upload features to HuggingFace
- `--hf-token`: HuggingFace API token (required if using --upload-hf)
- `--list-models`: List available models in the DeepJuice model zoo

### Examples

#### Using Multiple Models and Sampling Rates

```bash
# Try AlexNet and ResNet with different frame sampling rates (every 1st, 5th, and 10th frame)
python algonauts_feature_extraction.py --data-dir algonauts_data --models alexnet resnet50 --sampling-rates 1 5 10
```

#### List Available Models

```bash
python algonauts_feature_extraction.py --list-models
```

#### Using Datalad and HuggingFace Upload

```bash
python algonauts_feature_extraction.py --use-datalad --upload-hf --hf-token YOUR_HF_TOKEN
```

## Feature Extraction

The script extracts features from the final layers of the selected model, which typically have the highest performance for downstream tasks. For different model architectures, specific layers are selected:

- **AlexNet**: Classifier layers
- **ResNet**: Layer4 and fully connected layers
- **ViT**: Final transformer blocks and head layers
- **Other models**: Last 5 layers by default

Features are saved in H5 format, organized by subject, session, and run.

## Output Organization

Features are saved in the output directory with the following structure:
```
extracted_features/
  ├── model_name_sr1/          # Model with sampling rate 1
  │   └── model_name_sub-01_ses-task_run-01.h5
  ├── model_name_sr5/          # Model with sampling rate 5
  │   └── model_name_sub-01_ses-task_run-01.h5
  └── ...
```

## Uploading to HuggingFace

Features are uploaded to: https://huggingface.co/datasets/medarc/AlgonautsDS-features/tree/main

## Reference

This work is based on:
- Feature extraction code: https://www.kaggle.com/code/ccc3ready/feauture-extraction-latest/
- Dataset: https://www.kaggle.com/datasets/ckadirt/algonauts2025nsl
- DeepJuice: https://github.com/ColinConwell/DeepJuiceDev 