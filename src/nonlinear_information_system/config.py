"""
config.py
---------
Load and flatten YAML configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Candidate locations for the default config (checked in order).
_CANDIDATES = [
    Path(__file__).parent.parent.parent.parent / "config" / "default.yaml",  # editable install
    Path(__file__).parent.parent.parent / "config" / "default.yaml",          # wheel install
    Path("config") / "default.yaml",                                           # cwd fallback
]


def _default_config_path() -> Path:
    for p in _CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find config/default.yaml. "
        "Pass an explicit --config path or run from the project root."
    )


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
    config_path = Path(path) if path is not None else _default_config_path()
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
