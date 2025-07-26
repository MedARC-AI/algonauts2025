import argparse
import json
import math
import shutil
import time
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
    episode_filter,
)
from models import CrossSubjectConvLinearEncoder
from utils import pearsonr_score, get_sha

SUBJECTS = (1, 2, 3, 5)

ROOT = Path(__file__).parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CONFIG = ROOT / "config/default_cross_encoding.yaml"


def main(cfg: DictConfig):
    print("training cross-subject fmri encoder")

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

    print("creating model")
    model = CrossSubjectConvLinearEncoder(**cfg.model)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    param_count_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model:", model)
    print(
        f"num params (train): {param_count / 1e6:.2f} ({param_count_train / 1e6:.2f}M)"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_acc = None

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
            if name == cfg.val_set_name:
                val_acc = acc
            val_metrics[name] = metrics

        if epoch == 0 or val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            best_accs = val_accs
            best_metrics = val_metrics
            best_state = model.state_dict()
        else:
            # early stopping
            break

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
        print(json.dumps(best_accs), file=f)

    print("done! best accuracy:\n", json.dumps(best_accs))


def make_data_loaders(cfg: DictConfig) -> dict[str, DataLoader]:
    data_dir = Path(cfg.datasets_root or DEFAULT_DATA_DIR)
    data_dir = data_dir / "algonauts_2025.competitors"

    friends_fmri = load_algonauts2025_friends_fmri(data_dir, subjects=SUBJECTS)
    movie10_fmri = load_algonauts2025_movie10_fmri(data_dir, subjects=SUBJECTS)
    all_fmri = {**friends_fmri, **movie10_fmri}
    all_episodes = list(all_fmri)

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
            **ds_cfg,
        )

        batch_size = cfg.batch_size if ds_name == "train" else 1
        loader = DataLoader(dataset, batch_size=batch_size)

        data_loaders[ds_name] = loader

    return data_loaders


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
        sample = sample.to(device)
        batch_size = sample.size(0)
        data_time = time.monotonic() - end

        # forward pass
        output = model(sample)
        loss = F.mse_loss(output, sample)
        loss_item = loss.item()

        if math.isnan(loss_item) or math.isinf(loss_item):
            raise RuntimeError(
                "NaN/Inf loss encountered on step %d; exiting", batch_idx + 1
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
        sample = sample.to(device)
        batch_size = sample.size(0)

        # forward pass
        output = model(sample)
        loss = F.mse_loss(output, sample)
        loss_item = loss.item()

        loss_m.update(loss_item, batch_size)

        N, S, L, C = sample.shape
        assert N, S == (1, 4)
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
