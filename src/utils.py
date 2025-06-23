import inspect
import os
import subprocess
from typing import Any, Callable

import numpy as np


def pearsonr_score(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7
) -> np.ndarray:
    assert y_true.ndim == y_pred.ndim == 2

    y_true = y_true - y_true.mean(axis=0)
    y_true = y_true / (np.linalg.norm(y_true, axis=0) + eps)

    y_pred = y_pred - y_pred.mean(axis=0)
    y_pred = y_pred / (np.linalg.norm(y_pred, axis=0) + eps)

    score = (y_true * y_pred).sum(axis=0)
    return score


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def filter_kwargs(func: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    sigature = inspect.signature(func)
    kwargs = {k: v for k, v in kwargs.items() if k in sigature.parameters}
    return kwargs
