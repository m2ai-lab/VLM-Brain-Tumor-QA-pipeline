"""
config_utils.py — Shared configuration loader for QA-BrainTumor-VLM-UCSF.

Usage (in any script):
    from config_utils import load_config
    cfg = load_config()
    print(cfg["qa_path"])   # fully resolved absolute path

The loader:
  1. Finds config.yaml by walking up from this file's location.
  2. Reads the YAML into a flat dict.
  3. Iteratively resolves {variable} placeholders using Python's str.format_map
     until no further substitutions are possible (handles chained references).
  4. Returns the resolved dict.

config.yaml is gitignored. Users copy config.example.yaml → config.yaml and
fill in their paths. All other scripts can then call load_config() with no
additional arguments.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    print(
        "ERROR: PyYAML is not installed. Run:\n"
        "  pip install pyyaml\n"
        "or add it to your environment's requirements.",
        file=sys.stderr,
    )
    sys.exit(1)


def _find_config(start_dir: str, filename: str = "config.yaml") -> str:
    """Walk upward from start_dir until config.yaml is found."""
    current = os.path.abspath(start_dir)
    for _ in range(10):  # max 10 levels up
        candidate = os.path.join(current, filename)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        f"Could not find '{filename}' in '{start_dir}' or any parent directory.\n"
        "Have you copied config.example.yaml → config.yaml and filled it in?"
    )


def _resolve_placeholders(raw: dict[str, Any], max_passes: int = 10) -> dict[str, str]:
    """
    Iteratively expand {variable} placeholders in string values until stable.
    Non-string values (ints, bools, etc.) are left as-is.
    """
    resolved: dict[str, Any] = dict(raw)
    for _ in range(max_passes):
        changed = False
        for key, value in resolved.items():
            if not isinstance(value, str):
                continue
            try:
                new_value = value.format_map(resolved)
            except (KeyError, ValueError):
                continue  # unresolvable yet — try next pass
            if new_value != value:
                resolved[key] = new_value
                changed = True
        if not changed:
            break
    return resolved


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load and resolve config.yaml.

    Parameters
    ----------
    config_path : str | None
        Explicit path to the config file. If None, the file is auto-discovered
        by walking up from config_utils.py's location.

    Returns
    -------
    dict
        Fully resolved configuration dictionary.
    """
    if config_path is None:
        config_path = _find_config(os.path.dirname(os.path.abspath(__file__)))

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return _resolve_placeholders(raw)
