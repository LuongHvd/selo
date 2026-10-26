"""Run registry and provenance for the RMO branch.

Deliberately a **separate** registry file from `results/experiment_registry.csv`: the P0-P4 registry
has a fixed column set and must not be rewritten, while the RMO branch needs extra columns
(decision dimension D, scenario evaluations, scenario-bank seed).  The two histories coexist.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

RMO_REGISTRY = os.path.join("results", "rmo", "rmo_registry.csv")

FIELDS = [
    "phase", "run_id", "timestamp", "code_hash", "git_commit", "config_hash",
    "channel_seed", "optimizer_seed", "scenario_bank_seed",
    "eta", "M", "N", "J", "D", "method", "pop_size",
    "candidate_fe", "scenario_evals", "ao_calls", "runtime_s",
    "status", "error", "output_path", "host",
]

_CODE_HASH: Optional[str] = None


def code_hash(root: str = os.path.join("src", "risma")) -> str:
    """Content hash of the source tree.  This repository is not a git repo, so this stands in
    for the commit id; it is labelled as such in every registry row."""
    global _CODE_HASH
    if _CODE_HASH is not None:
        return _CODE_HASH
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            h.update(p.replace("\\", "/").encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    _CODE_HASH = h.hexdigest()[:16]
    return _CODE_HASH


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:      # noqa: BLE001
        return "not-a-git-repo"


def config_hash(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def machine_info() -> Dict[str, Any]:
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "hostname": socket.gethostname(), "cpu_count": os.cpu_count(),
            "code_hash": code_hash(), "git_commit": git_commit()}
    return info


class RmoRegistry:
    """Append-only CSV; failed runs stay registered with status=FAILED."""

    def __init__(self, path: str = RMO_REGISTRY):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        clean = {k: row.get(k, "") for k in FIELDS}
        clean["timestamp"] = clean["timestamp"] or time.strftime("%Y-%m-%dT%H:%M:%S")
        clean["host"] = clean["host"] or socket.gethostname()
        clean["code_hash"] = clean["code_hash"] or code_hash()
        clean["git_commit"] = clean["git_commit"] or git_commit()
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow(clean)


def save_rmo_run(out_dir: str, run_id: str, config: Dict[str, Any],
                 arrays: Dict[str, np.ndarray], scalars: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"{run_id}.npz")
    np.savez_compressed(npz, **{k: np.asarray(v) for k, v in arrays.items()})
    with open(os.path.join(out_dir, f"{run_id}.json"), "w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id, "config": config, "scalars": scalars,
                   "machine": machine_info()}, fh, indent=2, default=str)
    return npz
