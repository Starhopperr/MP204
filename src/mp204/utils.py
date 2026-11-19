"""Utility functions for MP204."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load settings and parameters from YAML file safely.

    Args:
        params_path: Relative or absolute path to params.yaml.

    Returns:
        Dict containing configured parameters.
    """
    path = Path(params_path)
    if not path.exists():
        # Fallback to root relative if called from inside a subpackage
        alt_path = Path(__file__).resolve().parent.parent.parent / "params.yaml"
        if alt_path.exists():
            path = alt_path

    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params
