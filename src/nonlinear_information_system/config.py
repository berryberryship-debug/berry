"""
config.py
---------
Load and flatten YAML configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML file.  Defaults to ``config/default.yaml`` in the
        project root when *None*.

    Returns
    -------
    dict
        Nested configuration dictionary with ``model`` and ``simulation``
        sub-dictionaries.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested YAML config into a single-level dict.

    The ``model`` and ``simulation`` sections are merged into one dict so
    that model functions can accept a single *cfg* argument.

    Parameters
    ----------
    cfg:
        Nested configuration as returned by :func:`load_config`.

    Returns
    -------
    dict
        Flat configuration dictionary.
    """
    flat: dict[str, Any] = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat
