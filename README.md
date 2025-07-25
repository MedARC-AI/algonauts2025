# MedARC Algonauts 2025 submission

Repository for MedARC group tackling the [Algonauts 2025 challenge](https://algonautsproject.com/).

## Installation

Clone the repository, [install uv](https://docs.astral.sh/uv/getting-started/installation/), and then run

```sh
uv sync
```

## Data

Download the [official challenge dataset](https://github.com/courtois-neuromod/algonauts_2025.competitors), and copy or link it to `datasets/algonauts_2025.competitors`.

## Features

Precomputed features extracted from a number of backbone models are available on [Huggingface](https://huggingface.co/datasets/medarc/AlgonautsDS-features). Once you have downloaded the features, copy or link them under `datasets/` like so

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
InternVL3_14B
├── InternVL3_14B/friends
│   ├── InternVL3_14B/friends/s1
│   │   ├── InternVL3_14B/friends/s1/friends_s01e01a.h5
│   │   ├── InternVL3_14B/friends/s1/friends_s01e01b.h5
...
```

## Training the default model

To train the model using the [default config](config/default_feature_encoding.yaml), run

```sh
uv run python train_feature_encoder.py
```
