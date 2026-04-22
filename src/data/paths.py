# src/data/paths.py
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """
    Root directory for datasets.

    Override with:
      export WAD_DATA_DIR=/abs/path/to/data
    """
    return Path(os.environ.get("WAD_DATA_DIR", repo_root() / "data")).resolve()


def raw_dir() -> Path:
    return data_root() / "raw"


def processed_dir() -> Path:
    return data_root() / "processed"
