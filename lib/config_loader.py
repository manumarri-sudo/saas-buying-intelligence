"""
Loads and validates the pipeline config.yaml.
Provides a single get_config() entry point used by all scripts.
"""

import os
import yaml
from pathlib import Path


_CONFIG_CACHE = None


def get_project_root() -> Path:
    """Return the project root (directory containing config.yaml)."""
    return Path(__file__).resolve().parent.parent


def get_config(config_path: str | None = None) -> dict:
    """Load config.yaml once and cache it."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    _CONFIG_CACHE = cfg
    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to project root."""
    return get_project_root() / relative


def _validate(cfg: dict) -> None:
    """Basic structural validation."""
    required_sections = [
        "project", "ingestion", "filtering", "extraction",
        "validation", "packaging", "embeddings", "rag",
    ]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")

    max_snippet = cfg["filtering"]["max_snippet_chars"]
    if max_snippet > 240:
        raise ValueError(
            f"max_snippet_chars={max_snippet} exceeds legal limit of 240"
        )
