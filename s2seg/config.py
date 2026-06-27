"""Tiny config loader: reads config.yaml into attribute-accessible namespaces."""

from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path=None):
    path = Path(path) if path else DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)
