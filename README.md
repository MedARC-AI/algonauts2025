# algonauts2025
Repository for MedARC group tackling Algonauts 2025 challenge

Link to Huggingface Dataset for storing extracted features:
https://huggingface.co/datasets/medarc/AlgonautsChallenge25/tree/main

## Setup

Initialize and update the submodules:

```bash
git submodule init
git submodule update
```

## Download features

To download developer_kit/stimulus_features (or replace with your desired directory):

```bash
git clone --no-checkout https://huggingface.co/datasets/medarc/AlgonautsDS-features
cd AlgonautsDS-features
git sparse-checkout init --cone
git sparse-checkout set developer_kit/stimulus_features
git checkout
```
