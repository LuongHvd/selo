"""Reproducibility infrastructure: master run registry + per-run provenance."""

from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

REGISTRY_FIELDS = [
    "experiment_id", "phase", "method", "seed", "M", "N", "J", "L", "A", "Pmax_dbm",
    "eta", "uncertainty_model", "S_design", "budget", "objective", "n_mc",
    "candidate_fe", "secrecy_evals", "ao_calls", "ao_iters",
    "runtime_s", "status", "error", "output_path", "timestamp", "host",
]


def machine_info() -> Dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:      # noqa: BLE001
        info["torch"] = None
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:      # noqa: BLE001
        info["git_commit"] = None
    return info


class Registry:
    """Append-only CSV registry; failed runs stay registered with status=FAILED."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=REGISTRY_FIELDS).writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        clean = {k: row.get(k, "") for k in REGISTRY_FIELDS}
        clean.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        clean["timestamp"] = clean["timestamp"] or time.strftime("%Y-%m-%dT%H:%M:%S")
        clean["host"] = clean["host"] or socket.gethostname()
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=REGISTRY_FIELDS).writerow(clean)


def save_run(out_dir: str, run_id: str, config: Dict[str, Any], arrays: Dict[str, np.ndarray],
             scalars: Dict[str, Any]) -> str:
    """Persist the raw result of one run.  Returns the .npz path."""
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"{run_id}.npz")
    np.savez_compressed(npz, **{k: np.asarray(v) for k, v in arrays.items()})
    with open(os.path.join(out_dir, f"{run_id}.json"), "w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id, "config": config, "scalars": scalars,
                   "machine": machine_info()}, fh, indent=2, default=str)
    return npz
