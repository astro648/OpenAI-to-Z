# Utilities for Earthdata login and GEDI downloads

from __future__ import annotations

import os
from pathlib import Path
import earthaccess

ROOT = Path(__file__).resolve().parent.parent


def _read_credential(name: str) -> str:
    env = os.getenv(f"EARTHDATA_{name.upper()}")
    if env:
        return env
    path = ROOT / f"{name}.txt"
    if path.exists():
        return path.read_text().strip()
    return ""


def login() -> None:
    """Authenticate with Earthdata using environment variables or text files."""
    username = _read_credential("username")
    password = _read_credential("password")
    try:
        earthaccess.login(username=username, password=password)
    except TypeError as exc:
        if "unexpected keyword argument" in str(exc):
            earthaccess.login(username, password)
        else:
            raise


def search(short_name: str, bbox: tuple[float, float, float, float]):
    return earthaccess.search_data(
        short_name=short_name,
        bounding_box=bbox,
        cloud_hosted=True,
    )


def download(granules, target: str | Path) -> None:
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)
    earthaccess.download(granules, str(target_path))

