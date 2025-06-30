# Utilities for Earthdata login and GEDI downloads

from __future__ import annotations

import os
from pathlib import Path
import earthaccess

ROOT = Path(__file__).resolve().parents[1]


def data_path(subfolder: str) -> Path:
    """Return the repository ``data/raw`` path joined with ``subfolder``."""
    return ROOT / "data" / "raw" / subfolder


def _read_credential(name: str) -> str:
    env = os.getenv(f"EARTHDATA_{name.upper()}")
    if env:
        return env
    path = ROOT / f"{name}.txt"
    if path.exists():
        return path.read_text().strip()
    return ""


def login() -> None:
    """Authenticate with Earthdata using stored credentials or fall back to interactive login."""
    username = _read_credential("username")
    password = _read_credential("password")
    if username and password:
        try:
            earthaccess.login(strategy="password", username=username, password=password)
            return
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Password login failed: {exc}\nFalling back to interactive login...")

    # If no credentials or password login failed, attempt interactive login
    earthaccess.login(strategy="interactive")


def search(short_name: str, bbox: tuple[float, float, float, float]):
    return earthaccess.search_data(
        short_name=short_name,
        bounding_box=bbox,
        cloud_hosted=True,
    )


from itertools import islice


def download(granules, target: str | Path, limit: int | None = None) -> None:
    """Download ``granules`` into ``target`` with an optional limit."""
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        granules = list(islice(granules, limit))

    earthaccess.download(granules, str(target_path))

