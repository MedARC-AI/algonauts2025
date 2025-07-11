import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from data import (
    Algonauts2025Dataset,
    load_merged_features,
    load_sharded_features,
    load_developer_features,
    episode_filter,
)
from models import MultiSubjectConvLinearEncoder
from transformer import Transformer
from conv1dnext import Conv1dNext
from utils import get_sha

SUBJECTS = (1, 2, 3, 5)

ROOT = Path(__file__).parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CONFIG = ROOT / "config/default_submission.yaml"


def main(cfg: DictConfig):
    print("generating submission predictions for multi-subject fmri encoder")

    sha_info = get_sha()
    print(sha_info)
    print("config:", OmegaConf.to_yaml(cfg), sep="\n")

    out_dir = Path(cfg.out_dir)
    prev_cfg = OmegaConf.load(out_dir / "config.yaml")
    prev_cfg['test_set_name'] = cfg['test_set_name']

    device = torch.device(cfg.device)
    print(f"running on: {device}")

    print("creating data loader")

    fmri_num_samples = load_fmri_num_samples(cfg)
    test_loader = make_data_loader(prev_cfg, fmri_num_samples)

    example_batch = next(iter(test_loader))
    # feat_dims = [feat.shape[-1] for feat in example_batch["features"]]
    feat_dims = [] # This is how it is done in the train_feature_encoder.py, let's see if this works
    for feat in example_batch["features"]:
        # features can be (N, T, C) or (N, T, L, C)
        dim = feat.shape[2] if feat.ndim == 3 else tuple(feat.shape[2:])
        feat_dims.append(dim)
    print("feat dims:", feat_dims)

    print("creating model")
    hidden_model_type = prev_cfg.model.pop("hidden_model")
    if hidden_model_type == "transformer":
        hidden_model_cfg = prev_cfg.transformer
        hidden_model = Transformer(
            embed_dim=prev_cfg.model.embed_dim, **hidden_model_cfg
        )
    elif hidden_model_type == "conv1dnext":
        hidden_model_cfg = prev_cfg.conv1dnext
        hidden_model = Conv1dNext(
            embed_dim=prev_cfg.model.embed_dim, **hidden_model_cfg
        )
    else:
        hidden_model = None

    model = MultiSubjectConvLinearEncoder(
        feat_dims=feat_dims,
        hidden_model=hidden_model,
        **prev_cfg.model,
    )
    print("model:", model)

    ckpt_path = out_dir / "ckpt.pt"
    print("loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)

    print("generating test predictions")

    submission_predictions = inference(
        model=model,
        test_loader=test_loader,
        test_num_samples=fmri_num_samples,
        device=device,
    )

    print_submission_summary(submission_predictions)

    print("saving test predictions")
    save_submission_predictions(submission_predictions, cfg.test_set_name, out_dir)

    print("done!")


def make_data_loader(
    cfg: DictConfig,
    fmri_num_samples: dict[str, int],
) -> DataLoader:
    # all_features = []
    # for feat_cfg in cfg.features:
    #     print(f"loading features:\n\n{OmegaConf.to_yaml(feat_cfg)}")
    #     features = load_features(cfg, **feat_cfg)
    #     all_features.append(features)
    all_features = []
    for feat_name in cfg.include_features:
        feat_cfg = cfg.features[feat_name]
        print(f"loading features {feat_name}:\n\n{OmegaConf.to_yaml(feat_cfg)}")
        features = load_features(cfg,**feat_cfg)
        all_features.append(features)

    all_episodes = list(all_features[0])

    if cfg.test_set_name == "friends-s7":
        filter_fn = episode_filter(seasons=[7], movies=[])
    elif cfg.test_set_name == "ood":
        filter_fn = episode_filter(seasons=[],movies=["chaplin","mononoke","passepartout","planetearth","pulpfiction","wot"])
    else:
        raise ValueError(f"test set {cfg.test_set_name} not implemented.")

    ds_episodes = list(filter(filter_fn, all_episodes))
    print(f"episodes: {cfg.test_set_name}:\n\n{ds_episodes}")

    fmri_num_samples = {
        ep: max(fmri_num_samples[sub][ep] for sub in fmri_num_samples)
        for ep in ds_episodes
    }

    dataset = Algonauts2025Dataset(
        episode_list=ds_episodes,
        feat_data=all_features,
        fmri_num_samples=fmri_num_samples,
        sample_length=None,
        shuffle=False,
    )

    loader = DataLoader(dataset, batch_size=1)

    return loader


MODEL_FEATURE_TYPES = {
    "internvl3_8b_8bit": "sharded",
    "whisper": "sharded",
    "internvl3_14b": "sharded",
    "InternVL3_8B_multiframe":"sharded",
    "qwen-2-5-omni-7b":"sharded",
    "qwen2-5_3B":"sharded",
    "dinov2":"sharded",
    "vjepa2_avg_feat":"sharded",
    "dinov2-giant":"sharded",
    "modernBert":"sharded",
    "meta-llama__Llama-3.2-1B": "merged",
    "MFCC":"developer",
    "slow_r50":"developer",
    "bert-base-uncased":"developer",
    "emonet":"onefile"
}


def load_features(
    cfg: DictConfig,
    model: str,
    layer: str,
    stem: str | None = None,
) -> dict[str, np.ndarray]:
    feat_type = MODEL_FEATURE_TYPES[model]

    data_dir = Path(cfg.datasets_root or DEFAULT_DATA_DIR)

    if feat_type == "sharded":
        assert stem is None, "stem not used"
        friends_features = load_sharded_features(
            data_dir / "features.sharded", model=model, layer=layer, series="friends"
        )
        movie10_features = load_sharded_features(
            data_dir / "features.sharded", model=model, layer=layer, series="movie10"
        )
        ood_features = load_sharded_features(
            data_dir / "features.sharded", model=model, layer=layer, series="ood"
        )
    elif feat_type == "developer":
        friends_features = load_developer_features(
            data_dir / "features.sharded", model=model, layer=layer, series="friends"
        )
        movie10_features = load_developer_features(
            data_dir / "features.sharded", model=model, layer=layer, series="movie10"
        )
        ood_features = load_developer_features(
            data_dir / "features.sharded", model=model, layer=layer, series="ood"
        )
    else:
        friends_features = load_merged_features(
            data_dir / "features.merged",
            model=model,
            layer=layer,
            series="friends",
            stem=stem,
        )
        movie10_features = load_merged_features(
            data_dir / "features.merged",
            model=model,
            layer=layer,
            series="movie10",
            stem=stem,
        )
        ood_features = load_merged_features(
            data_dir / "features.merged",
            model=model,
            layer=layer,
            series="ood",
            stem=stem,
        )
    features = {**friends_features, **movie10_features, **ood_features}
    return features


def load_fmri_num_samples(cfg: DictConfig):
    data_dir = Path(cfg.datasets_root or DEFAULT_DATA_DIR)

    fmri_num_samples_paths = sorted(
        (data_dir / "algonauts_2025.competitors/fmri").rglob(
            f"*_{cfg.test_set_name}_fmri_samples.npy"
        )
    )

    fmri_num_samples = {}
    for path in fmri_num_samples_paths:
        sub = path.parents[1].name
        fmri_num_samples[sub] = np.load(path, allow_pickle=True).item()

    return fmri_num_samples


@torch.no_grad()
def inference(
    *,
    model: torch.nn.Module,
    test_loader: DataLoader,
    test_num_samples: dict[str, dict[str, int]],
    device: torch.device,
):
    model.eval()

    submission_predictions = {f"sub-{subid:02d}": {} for subid in SUBJECTS}

    for batch_idx, batch in enumerate(test_loader):
        feats = batch["features"]
        episodes = batch["episode"]

        feats = [feat.to(device) for feat in feats]

        output = model(feats)
        N, S, L, C = output.shape
        assert N, S == (1, 4)

        output = output.cpu().numpy()

        for ii, episode in enumerate(episodes):
            for jj, subid in enumerate(SUBJECTS):
                sub = f"sub-{subid:02d}"
                pred = output[ii, jj]

                num_samples = test_num_samples[sub][episode]

                assert len(pred) >= num_samples
                pred = pred[:num_samples]

                submission_predictions[sub][episode] = pred

    return submission_predictions


def print_submission_summary(submission_predictions: dict[str, dict[str, np.ndarray]]):
    for subject, episodes_dict in submission_predictions.items():
        # Print the subject and episode number for Friends season 7
        print(f"Subject: {subject}")
        print(f"  Number of Episodes: {len(episodes_dict)}")
        # Print the predicted fMRI response shape for each episode
        for episode, predictions in episodes_dict.items():
            print(
                f"    - Episode: {episode}, Predicted fMRI shape: {predictions.shape}"
            )
        print("-" * 40)  # Separator for clarity


def save_submission_predictions(
    submission_predictions: dict[str, dict[str, np.ndarray]],
    test_set_name: str,
    out_dir: Path,
):
    # friends-s7 -> friends_s7
    test_set_name = test_set_name.replace("-", "_")

    output_file = out_dir / f"fmri_predictions_{test_set_name}.npy"
    np.save(output_file, submission_predictions)

    # Zip the saved file for submission
    zip_file = out_dir / f"fmri_predictions_{test_set_name}.zip"
    with zipfile.ZipFile(zip_file, "w") as zipf:
        zipf.write(output_file, output_file.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
