"""Load QHH runtime configuration with optional environment overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = Path(
    os.getenv("QHH_CONFIG_FILE", str(ROOT_DIR / "config.json"))
).expanduser()


def load_config() -> dict:
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


CONFIG = load_config()


def config_value(section: str, key: str, default=None):
    section_data = CONFIG.get(section, {})
    if not isinstance(section_data, dict):
        return default
    return section_data.get(key, default)


def env_or_config(env_name: str, section: str, key: str, default=None):
    value = os.getenv(env_name)
    if value is not None:
        return value
    return config_value(section, key, default)
