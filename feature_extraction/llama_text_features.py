import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from timm.utils import random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_extractor import FeatureExtractor
from utils import get_sha
from data import parse_friends_run, parse_movie10_run

ROOT = Path(__file__).parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CONFIG = ROOT / "config/default_text_features.yaml"

TR = 1.49
NUM_TOTAL_MOVIES = 395

# silent movie number of samples
CHAPLIN_NUM_SAMPLES = {
    "chaplin1": 432,
    "chaplin2": 405,
}

MOVIE_TITLES = {
    "bourne": "The Bourne Supremacy",
    "figures": "Hidden Figures",
    "life": "Life: Challenges of life, reptiles and amphibian mammals.",
    "wolf": "The Wolf of Wall Street",
    "chaplin": "The Pawnshop by Charlie Chaplin",
    "passepartout": "Passe-Partout - Épisode 1 - Bonjour",
    "planetearth": "Planet Earth: Mountains",
    "mononoke": "Princess Mononoke",
    "pulpfiction": "Pulp Fiction",
    "wot": "World of Tomorrow by Don Hertzfeldt",
}


def main(cfg: DictConfig):
    print("extracting text features")

    sha_info = get_sha()
    print(sha_info)
    print("config:", OmegaConf.to_yaml(cfg), sep="\n")

    cfg._sha = sha_info

    model_name = cfg.model.split("/")[-1]
    out_dir = Path(cfg.out_dir) / model_name
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

    paths = sorted((data_dir / "stimuli/transcripts").rglob("*.tsv"))
    # friends/s1/friends_s01e01a
    base_names = [str(p.relative_to(p.parents[2]).with_suffix("")) for p in paths]
    print(f"extracting features for {len(paths)} movies:\n\n{base_names}")
    assert sum(p.exists() for p in paths) == NUM_TOTAL_MOVIES

    transcript_tables = {
        name: pd.read_csv(path, sep="\t") for name, path in zip(base_names, paths)
    }
    # add dummy tables for silent chaplin movies, just for consistency
    for part, num_samples in CHAPLIN_NUM_SAMPLES.items():
        transcript_tables[f"ood/chaplin/ood_{part}"] = make_dummy_transcript_table(
            num_samples
        )

    print(f"loading model: {cfg.model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model)
    model = model.to(device)
    print(model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"num params: {param_count / 1e6:.2f}M")

    print(f"extracting from layers: {cfg.layers}")
    extractor = FeatureExtractor(model, cfg.layers)
    print(f"expanded layers: {extractor.expanded_layers}")

    extract_features(
        transcript_tables,
        out_dir=out_dir,
        tokenizer=tokenizer,
        extractor=extractor,
        device=device,
    )

    print("done!")


def make_dummy_transcript_table(num_rows: int):
    rows = [
        {
            "text_per_tr": np.nan,
            "words_per_tr": "[]",
            "onsets_per_tr": "[]",
            "durations_per_tr": "[]",
        }
        for ii in range(num_rows)
    ]
    table = pd.DataFrame.from_records(rows)
    return table


def format_text(base_name: str, transcript_table: pd.DataFrame):
    lines = [
        format_line(line, ii * TR)
        for ii, line in enumerate(transcript_table["text_per_tr"])
    ]
    text = "".join(lines)

    lengths = np.array([len(line) for line in lines])
    onsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    movie_name = get_movie_name(base_name)
    prefix = f"{movie_name}\n\nTranscript:\n\n"
    text = prefix + text
    onsets = len(prefix) + onsets
    return text, onsets


def format_line(line: str, secs: float) -> str:
    if line is np.nan:
        line = ""
    mins = int(secs) // 60
    secs = int(secs) % 60
    return f"[{mins:02d}:{secs:02d}] {line}\n"


def get_movie_name(base_name: str):
    series, group, run = base_name.split("/")
    run = run.split("_")[-1]
    if series == "friends":
        season, episode, part = parse_friends_run(run)
        movie_name = f"Friends S{season:02d}E{episode:02d} (Part {part.upper()})"
    else:
        movie, part = parse_movie10_run(run)
        movie_name = f"{MOVIE_TITLES[movie]} (Part {part})"
    return movie_name


def get_tr_token_onsets(token_offset_mapping: np.ndarray, tr_char_onsets: np.ndarray):
    tr_token_onsets = np.searchsorted(
        token_offset_mapping[:, 1] - 1, tr_char_onsets, side="left"
    )
    windows = np.diff(tr_token_onsets, append=[len(token_offset_mapping)])
    windows = np.maximum(windows, 1)
    tr_token_onsets = np.stack([tr_token_onsets, tr_token_onsets + windows], axis=1)
    return tr_token_onsets


@torch.no_grad()
def extract_features(
    transcript_tables: dict[str, pd.DataFrame],
    out_dir: Path,
    tokenizer: AutoTokenizer,
    extractor: FeatureExtractor,
    device: torch.device,
):
    for base_name, table in transcript_tables.items():
        # friends/s1/friends_s01e01a
        print(f"extracting {base_name}")
        out_path = out_dir / f"{base_name}.h5"

        text, onsets = format_text(base_name, table)

        token_dict = tokenizer(text, return_offsets_mapping=True, return_tensors="np")
        tr_token_onsets = get_tr_token_onsets(token_dict["offset_mapping"][0], onsets)

        input_ids = torch.from_numpy(token_dict["input_ids"]).to(device)

        with torch.autocast(device_type="cuda"):
            _, unpooled_features = extractor(input_ids)

        features = {}
        for layer, feat in unpooled_features.items():
            if isinstance(feat, tuple):
                feat = feat[0]
            feat = pool_tr_features(feat, tr_token_onsets).cpu().numpy()
            features[layer] = feat

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            for layer, feat in features.items():
                f[layer] = feat


def pool_tr_features(features: torch.Tensor, tr_token_onsets: np.ndarray):
    features = features.squeeze()
    N, C = features.shape
    T = len(tr_token_onsets)
    tr_features = torch.zeros(T, C, device=features.device, dtype=features.dtype)
    for ii in range(T):
        start_token, stop_token = tr_token_onsets[ii]
        assert stop_token > start_token
        tr_features[ii] = features[start_token:stop_token].mean(axis=0)
    return tr_features


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
