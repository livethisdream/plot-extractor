"""Settings from plot-extractor.env, overridable by environment variables.

Lookup order for each setting (first one that is set wins):
  1. command-line flag (handled by the caller)
  2. environment variable, e.g. PLOTX_REPLOT=false
  3. the settings file: ./plot-extractor.env, or the path in $PLOTX_CONFIG
  4. built-in default

The settings file is plain KEY=VALUE lines; '#' starts a comment and quotes
around values are optional. It is meant to be easy to edit from a phone.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_FILE = "plot-extractor.env"


def parse_env_file(text: str) -> dict[str, str]:
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, val = line.split("=", 1)
        val = val.strip()
        if val[:1] in ("'", '"') and val[-1:] == val[:1] and len(val) >= 2:
            val = val[1:-1]
        elif " #" in val:  # trailing comment on an unquoted value
            val = val.split(" #", 1)[0].rstrip()
        out[key.strip()] = val
    return out


@lru_cache(maxsize=4)
def _file_settings(path: str) -> dict[str, str]:
    p = Path(path)
    return parse_env_file(p.read_text()) if p.is_file() else {}


def config_path() -> str:
    return os.environ.get("PLOTX_CONFIG") or DEFAULT_FILE


def setting(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is not None and v.strip() != "":
        return v.strip()
    v = _file_settings(config_path()).get(name)
    if v is not None and v.strip() != "":
        return v.strip()
    return default


def flag(name: str, default: bool = True) -> bool:
    """On/off setting: false/0/no/off (any case) means off."""
    v = setting(name)
    if v is None:
        return default
    return v.lower() not in ("0", "false", "no", "off")
