
import re
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

SUBJECTS = (1, 2, 3, 5)

root_dir = Path("/home/mihir/projects")

data_dir = root_dir / "datasets" / "algonauts_2025.competitors"






def parse_friends_run(run: str):
    match = re.match(r"s([0-9]+)e([0-9]+)([a-z])", run)
    if match is None:
        raise ValueError(f"Invalid friends run {run}")

    season = int(match.group(1))
    episode = int(match.group(2))
    part = match.group(3)
    return season, episode, part

def parse_movie10_run(run: str):
    match = re.match(r"([a-z]+)([0-9]+)", run)
    if match is None:
        raise ValueError(f"Invalid movie run {run}")

    movie = match.group(1)
    part = int(match.group(2))
    return movie, part


def load_algonauts2025_friends_fmri(
    root: str | Path,
    subjects: list[int] | None = None,
    seasons: list[int] | None = None,
) -> dict[str, np.ndarray]:
    subjects = subjects or SUBJECTS
    seasons = seasons or list(range(1, 7))

    files = {
        sub: h5py.File(
            Path(root)
            / f"fmri/sub-{sub:02d}/func"
            / f"sub-{sub:02d}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
        )
        for sub in subjects
    }

    episode_key_maps = defaultdict(dict)
    seasons_set = set(seasons)
    for sub, file in files.items():
        for key in file.keys():
            entities = dict([ent.split("-", 1) for ent in key.split("_")])
            episode = entities["task"]
            season, _, _ = parse_friends_run(episode)
            if season in seasons_set:
                episode_key_maps[episode][sub] = key

    episode_list = sorted(
        [
            episode for episode, map in episode_key_maps.items()
            if len(map) == len(subjects)
        ]
    )

    data = {}
    for episode in episode_list:
        samples = []
        length = None
        for sub in subjects:
            key = episode_key_maps[episode][sub]
            sample = files[sub][key][:]
            sub_length = len(sample)
            samples.append(sample)
            length = min(length, sub_length) if length else sub_length
        data[episode] = np.stack([sample[:length] for sample in samples])
    
    return data


def load_algonauts2025_movie10_fmri(
    root: str | Path,
    subjects: list[int] | None = None,
    movies: list[str] | None = None,
    runs: list[int] | None = None,
) -> dict[str, np.ndarray]:
    subjects = subjects or SUBJECTS
    movies = movies or ["bourne", "wolf", "figures", "life"]
    runs = runs or [1, 2]

    files = {
        sub: h5py.File(
            Path(root)
            / f"fmri/sub-{sub:02d}/func"
            / f"sub-{sub:02d}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
        )
        for sub in subjects
    }

    episode_key_maps = defaultdict(dict)
    movies_set = set(movies)
    for sub, file in files.items():
        for key in file.keys():
            entities = dict([ent.split("-", 1) for ent in key.split("_")])
            episode = entities["task"]
            run = int(entities.get("run", 1))
            movie, _ = parse_movie10_run(episode)
            if movie in movies_set and run in runs:
                episode_key_maps[(episode, run)][sub] = key

    episode_list = sorted(
        [
            episode for episode, map in episode_key_maps.items()
            if len(map) == len(subjects)
        ]
    )

    data = {}
    for episode in episode_list:
        samples = []
        length = None
        for sub in subjects:
            key = episode_key_maps[episode][sub]
            sample = files[sub][key][:]
            sub_length = len(sample)
            samples.append(sample)
            length = min(length, sub_length) if length else sub_length
        data[episode] = np.stack([sample[:length] for sample in samples])
    
    return data

class Algonauts2025Dataset(IterableDataset):
    def __init__(
        self,
        fmri_data: dict[str, np.ndarray],
        feat_data: list[dict[str, np.ndarray]] | None = None,
        sample_length: int | None = 128,
        num_samples: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.fmri_data = fmri_data
        self.feat_data = feat_data

        self.episode_list = list(fmri_data)
        self.sample_length = sample_length
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        self._rng = np.random.default_rng(seed)
    
    def _iter_shuffle(self):
        sample_idx = 0
        while True:
            episode_order = self._rng.permutation(len(self.episode_list))

            for ii in episode_order:
                episode = self.episode_list[ii]
                feat_episode = episode[0] if isinstance(episode, tuple) else episode

                fmri = torch.from_numpy(self.fmri_data[episode]).float()
    
                if self.feat_data:
                    feats = [torch.from_numpy(data[feat_episode]).float() for data in self.feat_data]
                else:
                    feats = feat_samples = None

                # Nb, fmri and feature length often off by 1 or 2.
                # But assuming time locked to start.
                length = fmri.shape[1]
                if feats:
                    length = min(length, min(feat.shape[0] for feat in feats))

                if self.sample_length:
                    # Random segment of run
                    offset = int(self._rng.integers(0, length - self.sample_length + 1))
                    fmri_sample = fmri[:, offset: offset + self.sample_length]
                    if feats:
                        feat_samples = [
                            feat[offset: offset + self.sample_length] for feat in feats
                        ]
                else:
                    # Take full run
                    # Nb this only works for batch size 1 since runs are different length
                    fmri_sample = fmri[:, :length]
                    if feats:
                        feat_samples = [feat[:length] for feat in feats]

                if feat_samples:
                    yield episode, fmri_sample, feat_samples
                else:
                    yield episode, fmri_sample

                sample_idx += 1
                if self.num_samples and sample_idx >= self.num_samples:
                    return

    def _iter_ordered(self):
        sample_idx = 0
        for episode in self.episode_list:
            feat_episode = episode[0] if isinstance(episode, tuple) else episode
            fmri = torch.from_numpy(self.fmri_data[episode]).float()
            if self.feat_data:
                feats = [torch.from_numpy(data[feat_episode]).float() for data in self.feat_data]
            else:
                feats = feat_samples = None

            length = fmri.shape[1]
            if feats:
                length = min(length, min(feat.shape[0] for feat in feats))

            sample_length = self.sample_length or length

            for offset in range(0, length - sample_length + 1, sample_length):
                fmri_sample = fmri[:, offset: offset + sample_length]
                if feats:
                    feat_samples = [feat[offset: offset + sample_length] for feat in feats]

                if feat_samples:
                    yield episode, fmri_sample, feat_samples
                else:
                    yield episode, fmri_sample

                sample_idx += 1
                if self.num_samples and sample_idx >= self.num_samples:
                    return

    def __iter__(self):
        if self.shuffle:
            yield from self._iter_shuffle()
        else:
            yield from self._iter_ordered()


def load_medarc_features(
    root: str | Path,
    model: str,
    layer: str,
    series: str = "friends"
) -> dict[str, np.ndarray]:
    paths = sorted((Path(root) / model / series).rglob("*.h5"))

    features = {}
    for path in paths:
        episode = path.stem.split("_")[-1]  # friends_s01e01a, bourne01
        with h5py.File(path) as f:
            features[episode] = f[layer][:].squeeze()
    return features



def load_merged_features(
    path: str | Path,
    layer: str,
) -> dict[str, np.ndarray]:
    with h5py.File(path) as f:
        features = {k: f[k][layer][:] for k in f}
    return features


medarc_feature_root = root_dir / "features.medarc"
merged_feature_root = root_dir / "features.merged"

stimuli_features_friends = {}
stimuli_features_movie10 = {}
medarc_models_layers = [
    ("whisper", "layers.12.fc2"),
    ("whisper", "layers.31.fc2"),
    ("internvl3_8b_8bit", "language_model.model.layers.10.post_attention_layernorm"),
    ("internvl3_8b_8bit", "language_model.model.layers.20.post_attention_layernorm"),
]

for model, layer in medarc_models_layers:
    stimuli_features_friends[f"{model}/{layer}"] = load_medarc_features(
        medarc_feature_root, model=model, layer=layer, series="friends",
    )
    stimuli_features_movie10[f"{model}/{layer}"] = load_medarc_features(
        medarc_feature_root, model=model, layer=layer, series="movie10",
    )
merged_models_layers = [
    ("Llama-3.2-1B", "model.layers.7"),
    ("Llama-3.2-1B", "model.layers.15"),
]

for model, layer in merged_models_layers:
    # TODO: this path is awkward
    stimuli_features_friends[f"{model}/{layer}"] = load_merged_features(
        path=merged_feature_root / f"friends/meta-llama__{model}/context-long.h5",
        layer=layer,
    )
    stimuli_features_movie10[f"{model}/{layer}"] = load_merged_features(
        path=merged_feature_root / f"movie10/meta-llama__{model}/context-long.h5",
        layer=layer,
    )


friends_train_fmri = load_algonauts2025_friends_fmri(data_dir, seasons=range(1, 6))
friends_val_fmri = load_algonauts2025_friends_fmri(data_dir, seasons=[6])
movie10_test_fmri = load_algonauts2025_movie10_fmri(data_dir, runs=[1])
print(friends_train_fmri.keys())
print(friends_val_fmri.keys())
print(movie10_test_fmri.keys())

sample = friends_train_fmri["s01e05b"]
print("Sample shape (NTC):", sample.shape, sample.dtype)

dataset = Algonauts2025Dataset(
    movie10_test_fmri,
    list(stimuli_features_movie10.values()),
    sample_length=64,
    num_samples=10000,
    shuffle=True,
    seed=42,
)
total_bytes = 0
tic = time.monotonic()
for task, fmri_sample, feat_samples in tqdm(dataset):
    total_bytes += fmri_sample.numel() * 4
rt = time.monotonic() - tic
tput = total_bytes / 1024 ** 2 / rt 
print(f"run time={rt:.3f}s, MB/s={tput:.0f}")