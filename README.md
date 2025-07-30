# Predicting Brain Responses to Natural Movies with Multimodal LLMs

**Cesar Kadir Torrico Villanueva¹⁴\*, Jiaxin Cindy Tu¹²\*, Mihir Tripathy¹³\*, Connor Lane¹⁴\*, Rishab Iyer¹⁵, Paul S. Scotti¹⁴**

<br>

¹Medical AI Research Center (MedARC)  
²Psychological and Brain Sciences, Dartmouth College  
³Core for Advanced Magnetic Resonance Imaging (CAMRI), Baylor College of Medicine  
⁴Sophont  
⁵Princeton Neuroscience Institute

---

Repository for MedARC group tackling the [Algonauts 2025 challenge](https://algonautsproject.com/). 

[[Preprint](https://arxiv.org/abs/2507.19956)]

### Main Environment

For the main application, run the following command. This will create a default `.venv` and install the project's primary dependencies from the lock file.

```sh
uv sync
```

### Feature Extraction Environment

If you need to run the feature extraction scripts, they require a separate environment with different dependencies.

1.  **Create the environment** named `ext`:
    ```sh
    uv venv ext
    ```
2.  **Install its specific dependencies** using the `-p` flag to target the `ext` environment directly:
    ```sh
    uv pip install -r feat_ext_requirements.txt
    ```

## Data

Download the [official challenge dataset](https://github.com/courtois-neuromod/algonauts_2025.competitors), and copy or link it to `datasets/algonauts_2025.competitors`.

## Features

Precomputed features extracted from a number of backbone models are available on [Huggingface](https://huggingface.co/datasets/medarc/algonauts_2025.features). Once you have downloaded the features, copy or link them under `datasets/` like so

<!-- TODO: update with all features we used. -->

```
datasets/
├── algonauts_2025.competitors
└── features
    ├── InternVL3_14B
    ├── Llama-3.2-1B
    ├── Llama-3.2-3B
    ├── internvl3_8b_8bit
    ├── qwen-2-5-omni-7b
    ├── qwen2-5_3B
    ├── vjepa2_avg_feat
    └── whisper
```

Each feature directory should have a structure like

```
Llama-3.2-1B
├── friends
│   ├── s1
│   │   ├── friends_s01e01a.h5
│   │   ├── friends_s01e01b.h5
...
├── movie10
│   ├── bourne
│   │   ├── movie10_bourne01.h5
│   │   ├── movie10_bourne02.h5
...
└── ood
    ├── chaplin
    │   ├── ood_chaplin1.h5
    │   └── ood_chaplin2.h5
...
```

### Manually Extracting Features

If you want to extract features manually, the model-specific scripts are located in the [`feature_extraction/`](./feature_extraction/) directory.

To run an extraction script, follow these steps:

1.  **Activate the feature extraction environment**

    ```sh
    source ext/bin/activate
    ```

2.  **Run a script with its corresponding config**

    ```sh
    # This example is for the InternVL3 model
    python feature_extraction/internvl3_video_features.py --cfg-path=config/default_internvl3_features.yaml
    ```
## Training the default model

To train the model using the [default config](config/default_feature_encoding.yaml), run

```sh
uv run python train_feature_encoder.py
```

## Preparing submission

To prepare a submission to [codabench](https://www.codabench.org/competitions/9483/) for the OOD movies, run

```bash
uv run python submit_feature_encoder.py \
  --overrides \
  checkpoint_dir=output/feature_encoding_default \
  test_set_name=ood
```

To prepare a submission for Friends S7, set `test_set_name=friends-s7`. See the [submission config](config/default_submission.yaml) for more details.

## Training the ensemble

Generate the sweep of configs for training the ensemble

```bash
uv run generate_configs.py
```

Train all the models in the ensemble

```bash
bash scripts/run_multiple_config_training.sh
```

Ensemble the top predictions for each subject and parcel by running the notebook [prepare_stitching_submission.ipynb](prepare_stitching_submission.ipynb).
