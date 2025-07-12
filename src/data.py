"""Dataset utils for algonauts 2025 fmri and features."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import h5py
import torch
from torch.utils.data import IterableDataset

SUBJECTS = (1, 2, 3, 5)

class Algonauts2025Dataset(IterableDataset):
    def __init__(
        self,
        episode_list: list[str | tuple[str, int]],
        fmri_data: dict[str, np.ndarray] | None = None,
        feat_data: list[dict[str, np.ndarray]] | None = None,
        fmri_num_samples: dict[str, int] = None,
        sample_length: int | None = 128,
        num_samples: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        assert fmri_data or feat_data, "fmri or features required"

        # subset to requested episodes
        if fmri_data:
            fmri_data = {ep: fmri_data[ep] for ep in episode_list}

        if feat_data:
            # Debugging
            # for ii in range(len(feat_data)): 
                # print(f"feature:{ii}")
                # print(feat_data[ii].keys())
            feat_data = [
                {
                    # no run in the feature episodes.
                    ep: layer_feat_data[ep[0] if isinstance(ep, tuple) else ep]
                    for ep in episode_list
                }
                for layer_feat_data in feat_data
            ]

        if fmri_num_samples:
            fmri_num_samples = {ep: fmri_num_samples[ep] for ep in episode_list}

        self.episode_list = episode_list
        self.fmri_data = fmri_data
        self.feat_data = feat_data
        self.fmri_num_samples = fmri_num_samples

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

                # (subs, length, dim) and list of (length, dim)
                fmri, feats, length = self._get_fmri_feats(episode)

                if self.sample_length:
                    # Random segment of run
                    offset = self._rng.integers(0, length - self.sample_length + 1)
                    if fmri is not None:
                        fmri_sample = fmri[:, offset : offset + self.sample_length]
                    if feats is not None:
                        feat_samples = [
                            feat[offset : offset + self.sample_length] for feat in feats
                        ]
                else:
                    # Take full run
                    # Nb this only works for batch size 1 since runs are different length
                    if fmri is not None:
                        fmri_sample = fmri[:, :length]
                    if feats is not None:
                        feat_samples = [feat[:length] for feat in feats]

                if isinstance(episode, tuple):
                    episode, run = episode
                else:
                    run = 1

                sample = {"episode": episode, "run": run}
                if fmri is not None:
                    sample["fmri"] = fmri_sample
                if feats is not None:
                    sample["features"] = feat_samples

                yield sample

                sample_idx += 1
                if self.num_samples and sample_idx >= self.num_samples:
                    return

    def _iter_ordered(self):
        sample_idx = 0

        for episode in self.episode_list:
            # (subs, length, dim) and list of (length, dim)
            fmri, feats, length = self._get_fmri_feats(episode)

            if isinstance(episode, tuple):
                episode, run = episode
            else:
                run = 1

            sample_length = self.sample_length or length

            for offset in range(0, length - sample_length + 1, sample_length):
                if fmri is not None:
                    fmri_sample = fmri[:, offset : offset + sample_length]
                if feats:
                    feat_samples = [
                        feat[offset : offset + sample_length] for feat in feats
                    ]

                sample = {"episode": episode, "run": run}
                if fmri is not None:
                    sample["fmri"] = fmri_sample
                if feats is not None:
                    sample["features"] = feat_samples

                yield sample

                sample_idx += 1
                if self.num_samples and sample_idx >= self.num_samples:
                    return

    def _get_fmri_feats(
        self, episode: str | tuple[str, int]
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None, int]:
        if self.fmri_data:
            # shape (subs, length, dim)
            fmri = self.fmri_data[episode]
            fmri_length = fmri.shape[1]
        elif self.fmri_num_samples:
            fmri = None
            fmri_length = self.fmri_num_samples[episode]
        else:
            fmri = fmri_length = None

        if self.feat_data:
            # each shape (length, dim)
            feats = [data[episode] for data in self.feat_data]
            feat_length = max(len(feat) for feat in feats)
        else:
            feats = feat_length = None

        # Nb, fmri and feature length often off by 1 or 2.
        # But assuming time locked to start.
        length = fmri_length or feat_length
        feats = _pad_trunc_features(feats, length)

        if fmri is not None:
            fmri = torch.from_numpy(fmri).float()

        if feats is not None:
            feats = [torch.from_numpy(feat).float() for feat in feats]

        return fmri, feats, length

    def __iter__(self):
        if self.shuffle:
            yield from self._iter_shuffle()
        else:
            yield from self._iter_ordered()


def _pad_trunc_features(feats: list[np.ndarray], length: int) -> list[np.ndarray]:
    pad_trunc_feats = []
    for feat in feats:
        if len(feat) < length:
            padding = [(0, length - len(feat))] + (feat.ndim - 1) * [(0, 0)]
            feat = np.pad(feat, padding, mode="edge")
        else:
            feat = feat[:length]
        pad_trunc_feats.append(feat)
    return pad_trunc_feats


def load_algonauts2025_friends_fmri(
    root: str | Path,
    subjects: list[int] | None = None,
    seasons: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """load friends fmri data.

    returns a big dictionary mapping episode -> data. episode is like "s01e01a". the
    data are aligned and stacked across subjects, resulting in shape (subs, length, dim).

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


def parse_friends_run(run: str):
    match = re.match(r"s([0-9]+)e([0-9]+)([a-z])", run)
    if match is None:
        raise ValueError(f"Invalid friends run {run}")

    season = int(match.group(1))
    episode = int(match.group(2))
    part = match.group(3)
    return season, episode, part


def load_algonauts2025_movie10_fmri(
    root: str | Path,
    subjects: list[int] | None = None,
    movies: list[str] | None = None,
    runs: list[int] | None = None,
) -> dict[tuple[str, int], np.ndarray]:
    """load movie10 fmri data.

    returns a big dictionary mapping (episode, run) -> data. "episode" is movie and part
    like "bourne01". run refers to repeat run, 1 or 2. run defaults to 1 for movies
    without repeats. the data are aligned and stacked across subjects, resulting in
    shape (subs, length, dim).

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
    chaplin1 -> (chaplin, 1)
    """
    movie = "empty"
    part = 0

    match = re.fullmatch(r"([a-zA-Z]+)(\d+)", run)
    # match = re.match(r"([a-z]+)([0-9]+)", run)
    if match is None:
        print(f"Invalid movie run '{run}', returning empty list.")
        # raise ValueError(f"Invalid movie run {run}")
    else:
        movie = match.group(1)
        part = int(match.group(2))
    return movie, part


def load_sharded_features(
    root: str | Path, model: str, layer: str, series: str = "friends"
) -> dict[str, np.ndarray]:
    """Load features from h5 shards.

    This is what most people on the team make.

    todo: maybe take a list of features and then concatenate, to share projection across
    layers.
    """
    paths = sorted((Path(root) / model / series).rglob("*.h5"))

    features = {}
    for path in paths:
        stem_parts = path.stem.replace("-","_").split("_")
        episode = stem_parts[1] if len(stem_parts) > 1 else stem_parts[0] # friends_s01e01a, bourne01
        # episode = path.stem.split("_")[-1]  # friends_s01e01a, bourne01, task-chaplin1_video
        with h5py.File(path) as f:
            features[episode] = f[layer][:].squeeze()
    return features


def load_merged_features(
    root: str | Path,
    model: str,
    layer: str,
    series: str = "friends",
    stem: str | None = None,
) -> dict[str, np.ndarray]:
    """Load features from a merged h5 file.

    Connor makes these, bc he's annoying.
    """
    if stem is None:
        path = Path(root) / f"{series}/{model}.h5"
    else:
        path = Path(root) / f"{series}/{model}/{stem}.h5"

    with h5py.File(path) as f:
        features = {k: f[k][layer][:] for k in f}
    return features

def load_onefile_features(
    root: str | Path,
    model: str,
    layer: str,
    stem: str | None = None,
) -> dict[str, np.ndarray]:
    """Load features from all features in one h5 file.
    """
    if stem is None:
        path = Path(root) /f"{model}" / f"{model}.npy"
    else:
        path = Path(root) /f"{model}" /f"{model}{stem}.npy"

    features = np.load(path, allow_pickle=True).item()
    return features

def load_developer_features(
    root: str | Path,
    model: str,
    layer: str,
series: str = "friends"
) -> dict[str, np.ndarray]:
    all_h5_files = sorted((Path(root) / model).rglob("*/*.h5"))
    if series == "friends":
        keywords = {"friends"}
    elif series == "movie10":
        keywords = {"figures", "wolf", "life", "bourne"}
    else:
        keywords = {"chaplin","mononoke","passepartout","planetearth","pulpfiction","wot"}
    paths = [p for p in all_h5_files if any(p.name.startswith(kw) for kw in keywords)]
    features = {}
    for path in paths:
        episode = path.stem.split("_")[-3]  # s01e01a, bourne01    
        with h5py.File(path) as f:
            if episode in f.keys():
                tmp = f[f'{episode}/{layer}'][:].squeeze()
            else:
                tmp =f[f'{layer}'][:].squeeze()
            features[episode] = tmp.reshape(tmp.shape[0],-1)
    return features

def episode_filter(
    seasons: list[str] | None = None,
    movies: list[str] | None = None,
    runs: list[int] | None = None,
) -> Callable[[str | tuple[str, int]], bool]:
    seasons = set(seasons) if seasons is not None else set(range(1, 6))
    movies = set(movies) if movies is not None else {"bourne", "wolf"}
    runs = set(runs) if runs is not None else {1}

    def _filter(episode: str | tuple[str, int]) -> bool:
        if isinstance(episode, tuple):
            episode, run = episode
        else:
            run = 1

        if episode.startswith("s0"):
            season, _, _ = parse_friends_run(episode)
            if season not in seasons:
                return False
        else:
            movie, _ = parse_movie10_run(episode)
            if movie not in movies:
                return False

        if run not in runs:
            return False

        return True

    return _filter
