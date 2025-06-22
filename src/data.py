"""Dataset utils for algonauts 2025 fmri and features."""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import h5py
import torch
from torch.utils.data import IterableDataset

SUBJECTS = (1, 2, 3, 5)


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
                    feats = [
                        torch.from_numpy(data[feat_episode]).float()
                        for data in self.feat_data
                    ]
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
                    fmri_sample = fmri[:, offset : offset + self.sample_length]
                    if feats:
                        feat_samples = [
                            feat[offset : offset + self.sample_length] for feat in feats
                        ]
                else:
                    # Take full run
                    # Nb this only works for batch size 1 since runs are different length
                    fmri_sample = fmri[:, :length]
                    if feats:
                        feat_samples = [feat[:length] for feat in feats]

                shapes = [fmri_sample.shape] + [feat.shape for feat in feat_samples]
                print("shapes:", shapes)

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
                feats = [
                    torch.from_numpy(data[feat_episode]).float()
                    for data in self.feat_data
                ]
            else:
                feats = feat_samples = None

            length = fmri.shape[1]
            if feats:
                length = min(length, min(feat.shape[0] for feat in feats))

            sample_length = self.sample_length or length

            for offset in range(0, length - sample_length + 1, sample_length):
                fmri_sample = fmri[:, offset : offset + sample_length]
                if feats:
                    feat_samples = [
                        feat[offset : offset + sample_length] for feat in feats
                    ]

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


def parse_friends_run(run: str):
    match = re.match(r"s([0-9]+)e([0-9]+)([a-z])", run)
    if match is None:
        raise ValueError(f"Invalid friends run {run}")

    season = int(match.group(1))
    episode = int(match.group(2))
    part = match.group(3)
    return season, episode, part


def load_algonauts2025_friends_fmri(
    root: str | Path,
    subjects: list[int] | None = None,
    seasons: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """load friends fmri data.

    returns a big dictionary mapping episode -> data. episode is like "s01e01a".

    root: path to algonauts_2025.competitors directory
    """
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
            episode
            for episode, map in episode_key_maps.items()
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
) -> dict[tuple[str, int], np.ndarray]:
    """load movie10 fmri data.

    returns a big dictionary mapping (episode, run) -> data. "episode" is movie and part
    like "bourne01". run refers to repeat run, 1 or 2. run defaults to 1 for movies
    without repeats.

    root: path to algonauts_2025.competitors directory
    runs: which of repeat runs to include. subset of [1, 2]. not that if you pick 2,
        only movies with a second repeat will be included.
    """
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
            episode
            for episode, map in episode_key_maps.items()
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


def parse_movie10_run(run: str) -> tuple[str, int]:
    """
    bourne01 -> (bourne, 1)
    """
    match = re.match(r"([a-z]+)([0-9]+)", run)
    if match is None:
        raise ValueError(f"Invalid movie run {run}")

    movie = match.group(1)
    part = int(match.group(2))
    return movie, part


def load_sharded_features(
    root: str | Path, model: str, layer: str, series: str = "friends"
) -> dict[str, np.ndarray]:
    """Load features from h5 shards.

    This is what most people on the team make.
    """
    paths = sorted((Path(root) / model / series).rglob("*.h5"))

    features = {}
    for path in paths:
        episode = path.stem.split("_")[-1]  # friends_s01e01a, bourne01
        with h5py.File(path) as f:
            features[episode] = f[layer][:].squeeze()
    return features


def load_merged_features(path: str | Path, layer: str) -> dict[str, np.ndarray]:
    """Load features from a merged h5 file.

    Connor makes these, bc he's annoying.
    """
    with h5py.File(path) as f:
        features = {k: f[k][layer][:] for k in f}
    return features
