import argparse
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from timm.utils import AverageMeter, random_seed
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from data import (
    Algonauts2025Dataset,
    load_algonauts2025_friends_fmri,
    load_algonauts2025_movie10_fmri,
    load_merged_features,
    load_sharded_features,
    load_developer_features,
    load_onefile_features,
    episode_filter,
)
from models import MultiSubjectConvLinearEncoder
from transformer import Transformer
from conv1dnext import Conv1dNext
from utils import pearsonr_score, get_sha

SUBJECTS = (1, 2, 3, 5)

ROOT = Path(__file__).parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CONFIG = ROOT / "config/default_feature_encoding.yaml"

MODEL_FEATURE_TYPES = {
    "internvl3_8b_8bit": "sharded",
    "whisper": "sharded",
    "internvl3_14b": "sharded",
    "qwen-2-5-omni-7b":"sharded",
    "qwen2-5_3B":"sharded",
    "dinov2":"sharded",
    "vjepa2_avg_feat":"sharded",
    "dinov2-giant":"sharded",
    "modernBert":"sharded",
    "InternVL3_8B_multiframe":"sharded",
    "videomae2":"sharded",
    "Llama-3.2-1B": "sharded",
    "Llama-3.2-3B": "sharded",
    "meta-llama__Llama-3.2-1B": "merged",
    "MFCC":"developer",
    "slow_r50":"developer",
    "bert-base-uncased":"developer",
    "InternVL3_14B":"developer",
    "emonet":"onefile"
}

# Added a separate feature type mapping for submission as it is different for some models
SUBMISSION_MODEL_FEATURE_TYPES = {
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
    "videomae2":"sharded",
    "Llama-3.2-1B": "sharded",
    "Llama-3.2-3B": "sharded",
    "meta-llama__Llama-3.2-1B": "merged",
    "MFCC":"developer",
    "slow_r50":"developer",
    "bert-base-uncased":"developer",
    "InternVL3_14B":"sharded", # Different from training
    "emonet":"onefile"
}


def main(cfg: DictConfig):
    print("training multi-subject fmri encoder")

    sha_info = get_sha()
    print(sha_info)
    print("config:", OmegaConf.to_yaml(cfg), sep="\n")

    cfg._sha = sha_info

    out_dir = Path(cfg.out_dir)
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

    print("creating data loaders")
    data_loaders = make_data_loaders(cfg)
    train_loader = data_loaders["train"]
    val_loaders = data_loaders.copy()
    val_loaders.pop("train")

    batch = next(iter(train_loader))

    feat_dims = []
    for feat in batch["features"]:
        dim = feat.shape[2] if feat.ndim == 3 else tuple(feat.shape[2:])
        feat_dims.append(dim)

    print("feat dims:", feat_dims)

    print("creating model")
    hidden_model_type = cfg.model.pop("hidden_model")
    if hidden_model_type == "transformer":
        hidden_model_cfg = cfg.transformer
        hidden_model = Transformer(embed_dim=cfg.model.embed_dim, **hidden_model_cfg)
    elif hidden_model_type == "conv1dnext":
        hidden_model_cfg = cfg.conv1dnext
        hidden_model = Conv1dNext(embed_dim=cfg.model.embed_dim, **hidden_model_cfg)
    else:
        hidden_model = None

    model = MultiSubjectConvLinearEncoder(
        feat_dims=feat_dims,
        hidden_model=hidden_model,
        **cfg.model,
    )
    print("model:", model)

    if cfg.checkpoint:
        print("loading checkpoint:", cfg.checkpoint)
        ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt["model"],
            strict=False,
        )
        print("missing_keys:", missing_keys)
        print("unexpected:", unexpected_keys)

    if cfg.freeze_decoder:
        for p in model.shared_decoder.parameters():
            p.requires_grad_(False)
        for p in model.subject_decoders.parameters():
            p.requires_grad_(False)

    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    param_count_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"num params (train): {param_count / 1e6:.2f} ({param_count_train / 1e6:.2f}M)"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_acc = None
    best_state = None

    for epoch in range(cfg.epochs):
        train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_acc = None
        val_accs = {}
        val_metrics = {}
        for ii, (name, loader) in enumerate(val_loaders.items()):
            acc, metrics = evaluate(
                epoch=epoch,
                model=model,
                val_loader=loader,
                device=device,
                ds_name=name,
            )
            val_accs[name] = acc
            if ii == 0:
                val_acc = acc
            val_metrics[name] = metrics

        if best_acc is None or val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            best_accs = val_accs
            best_metrics = val_metrics
            best_state = model.state_dict()
        else:
            # early stopping
            break

    if best_state is not None:
        with (out_dir / "ckpt.pt").open("wb") as f:
            torch.save(
                {
                    "cfg": OmegaConf.to_container(cfg),
                    "model": best_state,
                    "epoch": best_epoch,
                    "acc": best_acc,
                    "accs": best_accs,
                    "metrics": best_metrics,
                },
                f,
            )

        with (out_dir / "acc.json").open("w") as f:
            print(json.dumps(best_epoch), file=f)
            print(json.dumps(best_accs), file=f)

        print("done! best accuracy:\n", json.dumps(best_accs))

        # Run inference on OOD dataset
        if cfg.run_ood_prediction:
            print("\n" + "="*80)
            print("Running inference on OOD dataset")
            print("="*80 + "\n")
            run_ood_inference(cfg, out_dir, device)
    else:
        print("Training did not improve, no checkpoint saved.")


def make_data_loaders(cfg: DictConfig) -> dict[str, DataLoader]:
    print("loading fmri data")

    data_dir = Path(cfg.datasets_root or DEFAULT_DATA_DIR)

    friends_fmri = load_algonauts2025_friends_fmri(
        data_dir / "algonauts_2025.competitors", subjects=SUBJECTS
    )
    movie10_fmri = load_algonauts2025_movie10_fmri(
        data_dir / "algonauts_2025.competitors", subjects=SUBJECTS
    )
    all_fmri = {**friends_fmri, **movie10_fmri}
    all_episodes = list(all_fmri)

    all_features = []
    for feat_name in cfg.include_features:
        feat_cfg = cfg.features[feat_name]
        print(f"loading features {feat_name}:\n\n{OmegaConf.to_yaml(feat_cfg)}")
        features = load_features(cfg, **feat_cfg)
        all_features.append(features)

    data_loaders = {}

    for ds_name, ds_cfg in cfg.datasets.items():
        print(f"loading dataset: {ds_name}\n\n{OmegaConf.to_yaml(ds_cfg)}")

        ds_cfg = ds_cfg.copy()
        filter_cfg = ds_cfg.pop("filter")
        filter_fn = episode_filter(**filter_cfg)
        ds_episodes = list(filter(filter_fn, all_episodes))
        print(f"episodes: {ds_name}:\n\n{ds_episodes}")

        dataset = Algonauts2025Dataset(
            episode_list=ds_episodes,
            fmri_data=all_fmri,
            feat_data=all_features,
            **ds_cfg,
        )

        batch_size = cfg.batch_size if ds_name == "train" else 1
        loader = DataLoader(dataset, batch_size=batch_size)

        data_loaders[ds_name] = loader

    return data_loaders


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
    elif feat_type == "developer":
        assert stem is None, "stem not used"
        friends_features = load_developer_features(
            data_dir / "features.sharded", model=model, layer=layer, series="friends"
        )
        movie10_features = load_developer_features(
            data_dir / "features.sharded", model=model, layer=layer, series="movie10"
        )
    elif feat_type == "onefile":
        features = load_onefile_features(
            data_dir / "features.onefile",
            model=model,
            layer=layer,
            stem=stem,
        )
        return features # Return early as onefile features are structured differently
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
    
    features = {**friends_features, **movie10_features}
    return features


def train_one_epoch(
    *,
    epoch: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    loss_m = AverageMeter()
    data_time_m = AverageMeter()
    step_time_m = AverageMeter()

    end = time.monotonic()
    for batch_idx, batch in enumerate(train_loader):
        sample = batch["fmri"]
        feats = batch["features"]
        sample = sample.to(device)
        feats = [feat.to(device) for feat in feats]
        batch_size = sample.size(0)
        data_time = time.monotonic() - end

        # forward pass
        output = model(feats)
        loss = F.mse_loss(output, sample)
        loss_item = loss.item()

        if math.isnan(loss_item) or math.isinf(loss_item):
            raise RuntimeError(
                f"NaN/Inf loss encountered on step {batch_idx + 1}; exiting"
            )

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # end of iteration timing
        if use_cuda:
            torch.cuda.synchronize()
        step_time = time.monotonic() - end

        loss_m.update(loss_item, batch_size)
        data_time_m.update(data_time, batch_size)
        step_time_m.update(step_time, batch_size)

        if (batch_idx + 1) % 20 == 0:
            tput = batch_size / step_time_m.avg
            if use_cuda:
                alloc_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                res_mem_gb = torch.cuda.max_memory_reserved() / 1e9
            else:
                alloc_mem_gb = res_mem_gb = 0.0

            print(
                f"Train: {epoch:>3d} [{batch_idx:>3d}]"
                f"  Loss: {loss_m.val:#.3g} ({loss_m.avg:#.3g})"
                f"  Time: {data_time_m.avg:.3f},{step_time_m.avg:.3f} {tput:.0f}/s"
                f"  Mem: {alloc_mem_gb:.2f},{res_mem_gb:.2f} GB"
            )

        # Restart timer for next iteration
        end = time.monotonic()


@torch.no_grad()
def evaluate(
    *,
    epoch: int,
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ds_name: str = "val",
):
    model.eval()

    loss_m = AverageMeter()

    samples = []
    outputs = []

    for batch_idx, batch in enumerate(val_loader):
        sample = batch["fmri"]
        feats = batch["features"]
        sample = sample.to(device)
        feats = [feat.to(device) for feat in feats]
        batch_size = sample.size(0)

        # forward pass
        output = model(feats)
        loss = F.mse_loss(output, sample)
        loss_item = loss.item()

        loss_m.update(loss_item, batch_size)

        N, S, L, C = sample.shape
        assert N == 1 and S == 4, f"Unexpected shape: {sample.shape}"
        samples.append(sample.cpu().numpy().swapaxes(0, 1).reshape((S, N * L, C)))
        outputs.append(output.cpu().numpy().swapaxes(0, 1).reshape((S, N * L, C)))

    # (S, N, C)
    samples = np.concatenate(samples, axis=1)
    outputs = np.concatenate(outputs, axis=1)

    metrics = {}

    # Encoding accuracy metrics
    dim = samples.shape[-1]
    acc = 0.0
    acc_map = np.zeros(dim)
    for ii, sub in enumerate(SUBJECTS):
        y_true = samples[ii].reshape(-1, dim)
        y_pred = outputs[ii].reshape(-1, dim)
        metrics[f"accmap_sub-{sub}"] = acc_map_i = pearsonr_score(y_true, y_pred)
        metrics[f"acc_sub-{sub}"] = acc_i = np.mean(acc_map_i)
        acc_map += acc_map_i / len(SUBJECTS)
        acc += acc_i / len(SUBJECTS)

    metrics["accmap_avg"] = acc_map
    metrics["acc_avg"] = acc
    accs_fmt = ",".join(
        f"{val:.3f}" for key, val in metrics.items() if key.startswith("acc_sub-")
    )

    print(
        f"Evaluate ({ds_name}): {epoch:>3d}"
        f"  Loss: {loss_m.avg:#.3g}"
        f"  Acc: {accs_fmt} ({acc:.3f})"
    )

    return acc, metrics

# --- Functions for OOD Inference ---

def run_ood_inference(cfg: DictConfig, out_dir: Path, device: torch.device):
    """
    Main function to run inference on the OOD dataset.
    """
    print("creating ood data loader")
    fmri_num_samples = load_fmri_num_samples(cfg, "ood")
    test_loader = make_ood_data_loader(cfg, fmri_num_samples)

    example_batch = next(iter(test_loader))
    feat_dims = []
    for feat in example_batch["features"]:
        dim = feat.shape[2] if feat.ndim == 3 else tuple(feat.shape[2:])
        feat_dims.append(dim)
    print("feat dims for ood:", feat_dims)

    print("re-creating model for inference")
    # We pop 'hidden_model' from cfg, so we need a copy
    model_cfg = cfg.model.copy()
    try:
        hidden_model_type = model_cfg.pop("hidden_model")
    except KeyError:
        hidden_model_type = "none"
    if hidden_model_type == "transformer":
        hidden_model_cfg = cfg.transformer
        hidden_model = Transformer(
            embed_dim=model_cfg['embed_dim'], **hidden_model_cfg
        )
    elif hidden_model_type == "conv1dnext":
        hidden_model_cfg = cfg.conv1dnext
        hidden_model = Conv1dNext(
            embed_dim=model_cfg['embed_dim'], **hidden_model_cfg
        )
    else:
        hidden_model = None

    model = MultiSubjectConvLinearEncoder(
        feat_dims=feat_dims,
        hidden_model=hidden_model,
        **model_cfg,
    )

    ckpt_path = out_dir / "ckpt.pt"
    print("loading best checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)

    print("generating ood predictions")
    submission_predictions = inference(
        model=model,
        test_loader=test_loader,
        test_num_samples=fmri_num_samples,
        device=device,
    )

    print_submission_summary(submission_predictions)

    print("saving ood predictions")
    save_submission_predictions(submission_predictions, "ood", out_dir)

    print("ood inference done!")


def make_ood_data_loader(
    cfg: DictConfig,
    fmri_num_samples: dict[str, int],
) -> DataLoader:
    all_features = []
    for feat_name in cfg.include_features:
        feat_cfg = cfg.features[feat_name]
        print(f"loading submission features {feat_name}:\n\n{OmegaConf.to_yaml(feat_cfg)}")
        features = load_submission_features(cfg, **feat_cfg)
        all_features.append(features)

    all_episodes = list(all_features[0])

    filter_fn = episode_filter(seasons=[], movies=["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"])
    
    ds_episodes = list(filter(filter_fn, all_episodes))
    print(f"OOD episodes:\n\n{ds_episodes}")

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


def load_submission_features(
    cfg: DictConfig,
    model: str,
    layer: str,
    stem: str | None = None,
) -> dict[str, np.ndarray]:
    feat_type = SUBMISSION_MODEL_FEATURE_TYPES[model]

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
    else: # merged
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


def load_fmri_num_samples(cfg: DictConfig, test_set_name: str):
    data_dir = Path(cfg.datasets_root or DEFAULT_DATA_DIR)

    fmri_num_samples_paths = sorted(
        (data_dir / "algonauts_2025.competitors/fmri").rglob(
            f"*_{test_set_name}_fmri_samples.npy"
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
        assert N == 1 and S == 4, f"Unexpected shape: {output.shape}"

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
        print(f"Subject: {subject}")
        print(f"  Number of Episodes: {len(episodes_dict)}")
        for episode, predictions in episodes_dict.items():
            print(
                f"    - Episode: {episode}, Predicted fMRI shape: {predictions.shape}"
            )
        print("-" * 40)


def save_submission_predictions(
    submission_predictions: dict[str, dict[str, np.ndarray]],
    test_set_name: str,
    out_dir: Path,
):
    test_set_name = test_set_name.replace("-", "_")

    output_file = out_dir / f"fmri_predictions_{test_set_name}.npy"
    np.save(output_file, submission_predictions)

    zip_file = out_dir / f"fmri_predictions_{test_set_name}.zip"
    with zipfile.ZipFile(zip_file, "w") as zipf:
        zipf.write(output_file, output_file.name)
    print(f"Submission file saved at: {zip_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    # Added argument to control OOD prediction
    parser.add_argument("--run_ood_prediction", action="store_true",
                        help="Run inference on the OOD set after training.")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    
    # Add the command line argument to the config
    cfg.run_ood_prediction = args.run_ood_prediction
    
    main(cfg)