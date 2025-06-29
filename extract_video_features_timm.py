import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import timm
import timm.data
import torch
from omegaconf import DictConfig, OmegaConf
from timm.utils import random_seed
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from feature_extractor import FeatureExtractor, FeatureAdaptiveAvgPool2d
from video_utils import read_video
from utils import get_sha

ROOT = Path(__file__).parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CONFIG = ROOT / "config/default_video_features_timm.yaml"

TR = 1.49
NUM_TOTAL_MOVIES = 385


def main(cfg: DictConfig):
    print("extracting timm video features")

    sha_info = get_sha()
    print(sha_info)
    print("config:", OmegaConf.to_yaml(cfg), sep="\n")

    cfg._sha = sha_info

    out_dir = Path(cfg.out_dir) / cfg.model
    if out_dir.exists():
        if not cfg.overwrite:
            print(f"output {out_dir} exists; exiting.")
            return
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    random_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"running on: {device}")

    data_dir = Path(cfg.datasets_root or DEFAULT_DATA_DIR)
    data_dir = data_dir / "algonauts_2025.competitors"

    paths = sorted((data_dir / "stimuli/movies").rglob("*.mkv"))
    # exclude git annex files, hack
    paths = [p for p in paths if not p.name.startswith("MD5E-")]

    episodes = [p.stem for p in paths]
    print(f"extracting features for {len(paths)} movies:\n\n{episodes}")
    assert sum(p.exists() for p in paths) == NUM_TOTAL_MOVIES

    print(f"loading model: {cfg.model}")
    model, data_config, transforms = create_timm_model(cfg)
    model = model.to(device)
    print(model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"num params: {param_count / 1e6:.2f}M")

    print(f"extracting from layers: {cfg.layers}")
    extractor = FeatureExtractor(model, cfg.layers)
    print(f"expanded layers: {extractor.expanded_layers}")

    pool_fn = FeatureAdaptiveAvgPool2d(
        output_size=cfg.grid_size,
        num_prefix_tokens=model.num_prefix_tokens,
    )
    print(f"pooling: {pool_fn}")

    dataset = VideoDataset(paths, transforms, interval=TR)
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    extract_features(
        loader=loader,
        out_dir=out_dir,
        extractor=extractor,
        pool_fn=pool_fn,
        batch_size=cfg.batch_size,
        device=device,
    )

    print("done!")


def create_timm_model(cfg: DictConfig):
    model = timm.create_model(
        cfg.model,
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    return model, data_config, transforms


@torch.no_grad()
def extract_features(
    loader: DataLoader,
    out_dir: Path,
    extractor: FeatureExtractor,
    pool_fn: Callable,
    batch_size: int,
    device: torch.device,
):
    for path, images in loader:
        # movies/friends/s1/friends_s01e01a.mkv
        base_path = path.relative_to(path.parents[2])
        print(f"extracting {base_path}")

        out_path = out_dir / base_path.with_suffix(".h5")

        features = defaultdict(list)

        for ii in tqdm(range(0, len(images), batch_size)):
            batch = images[ii : ii + batch_size]
            batch = batch.to(device)

            with torch.autocast(device_type="cuda"):
                batch_embedding, batch_features = extractor(batch)

            features["embedding"].append(batch_embedding.cpu().numpy())
            for layer, feat in batch_features.items():
                feat = pool_fn(feat)
                feat = feat.cpu().numpy()
                if feat.shape[1] == 1:
                    feat = feat.squeeze(1)
                features[layer].append(feat)

        features = {k: np.concatenate(v) for k, v in features.items()}

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as f:
            for layer, feat in features.items():
                f[layer] = feat


class VideoDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        transform: Callable,
        interval: float,
    ):
        super().__init__()
        self.paths = paths
        self.transform = transform
        self.interval = interval

    def __getitem__(self, index):
        path = self.paths[index]
        images = []
        for _, img in read_video(path, interval=self.interval):
            img = self.transform(img)
            images.append(img)
        images = torch.stack(images)
        return path, images

    def __len__(self):
        return len(self.paths)


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
