import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from data import (
    Algonauts2025Dataset,
    load_sharded_features,
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

MODELS_DICT = {
    "multi_sub_conv_linear": MultiSubjectConvLinearEncoder,
}


def main(cfg: DictConfig):
    print("generating submission predictions for multi-subject fmri encoder")

    sha_info = get_sha()
    print(sha_info)
    print("config:", OmegaConf.to_yaml(cfg), sep="\n")

    ckpt_dir = Path(cfg.checkpoint_dir)
    prev_cfg = OmegaConf.load(ckpt_dir / "config.yaml")

    out_dir = ckpt_dir / f"submit_{cfg.test_set_name}"
    if out_dir.exists():
        if not cfg.overwrite:
            print(f"output {out_dir} exists; exiting.")
            return
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)

    device = torch.device(cfg.device)
    print(f"running on: {device}")

    print("creating data loader")

    fmri_num_samples = load_fmri_num_samples(cfg)
    test_loader = make_data_loader(cfg, prev_cfg, fmri_num_samples)

    batch = next(iter(test_loader))

    feat_dims = []
    for feat in batch["features"]:
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

    ckpt_path = ckpt_dir / "ckpt.pt"
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
    prev_cfg: DictConfig,
    fmri_num_samples: dict[str, int],
) -> DataLoader:
    all_features = []
    for feat_name in prev_cfg.include_features:
        model, layer = feat_name.split("/")
        feat_cfg = prev_cfg.features[model]
        model_name = feat_cfg.model
        layer_name = feat_cfg.layers[layer]
        print(f"loading features {feat_name} ({model_name}/{layer_name})")
        features = load_features(prev_cfg, model_name, layer_name)

        # pre-pool features if we are doing average pooling, to save space and time.
        if prev_cfg.model.global_pool == "avg":
            features = pool_features(features)

        all_features.append(features)

    all_episodes = list(all_features[0])

    if cfg.test_set_name == "friends-s7":
        filter_fn = episode_filter(seasons=[7], movies=[])
    elif cfg.test_set_name == "ood":
        filter_fn = episode_filter(
            seasons=[],
            movies=[
                "chaplin",
                "mononoke",
                "passepartout",
                "planetearth",
                "pulpfiction",
                "wot",
            ],
        )
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


def pool_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pooled = {}
    for key, feat in features.items():
        assert feat.ndim in {2, 3}
        if feat.ndim == 3:
            feat = feat.mean(axis=1)
        pooled[key] = feat
    return pooled


def load_features(cfg: DictConfig, model: str, layer: str) -> dict[str, np.ndarray]:
    data_dir = Path(cfg.datasets_root or DEFAULT_DATA_DIR)
    # TODO: we only really need to load friends s7 or ood, depending on the test set.
    friends_features = load_sharded_features(
        data_dir / "features", model=model, layer=layer, series="friends"
    )
    ood_features = load_sharded_features(
        data_dir / "features", model=model, layer=layer, series="ood"
    )
    features = {**friends_features, **ood_features}
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
