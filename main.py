"""Process GEDI Lidar tiles and query OpenAI for archaeological clues.

Each ``.h5`` file under ``data/raw`` is converted into a high‑contrast
false‑color PNG image saved in subdirectories of ``data/processed`` that
mirror the raw folder name (e.g. ``data/processed/serra`` for
``data/raw/serra``). The image is analysed using
the OpenAI API and results above a confidence of ``6`` are stored in CSV files
under the ``results`` directory. The CSVs contain the columns ``id``, ``pros``,
``cons`` and ``confidence``.

The script is intentionally verbose with logging to aid in tracing processing
and troubleshooting malformed files. The OpenAI API key is expected in a file
named ``openai_api_key.txt`` located at the repository root.
"""

import re
import json
import logging
import os
from pathlib import Path
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import base64

import h5py
import numpy as np
from PIL import Image
from skimage import exposure
from scipy.interpolate import griddata
import openai
import csv

# Global OpenAI client reused across requests
openai_client: typing.Optional[openai.OpenAI] = None

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Path to API key file located at repository root
API_KEY_FILE = Path(__file__).resolve().parent / "openai_api_key.txt"


def load_openai_key() -> str:
    """Read the OpenAI API key from ``openai_api_key.txt`` if available."""
    try:
        key = API_KEY_FILE.read_text().strip()
        if key:
            openai.api_key = key
            global openai_client
            openai_client = openai.OpenAI(api_key=key)
            logging.info("Loaded OpenAI API key from %s", API_KEY_FILE)
            return key
        logging.warning("OpenAI API key file %s is empty", API_KEY_FILE)
    except FileNotFoundError:
        logging.warning("OpenAI API key file %s not found", API_KEY_FILE)
    return ""

def find_tile_id(filename: str) -> str:
    """Extract the five-digit tile id from a GEDI file name."""
    match = re.search(r'(T\d{5})', filename)
    if match:
        return match.group(1)
    raise ValueError(f"No tile id found in {filename}")


def safe_cast(value: typing.Any, to_type: typing.Callable, default: typing.Any) -> typing.Any:
    """Safely cast ``value`` to ``to_type`` returning ``default`` on failure."""
    try:
        return to_type(value)
    except (ValueError, TypeError):
        try:
            if to_type is int:
                return int(float(value))
        except Exception:
            pass
        return default


def _rasterize_gedi_tiles(
    f: h5py.File,
    grid_res: float = 0.001,
    tile_px: int = 512,
    max_dim: int = 2048,
) -> list[np.ndarray]:
    """Rasterize GEDI point clouds and split long swaths into square tiles.

    ``GEDI`` footprints frequently form very long, thin tracks.  Instead of
    cropping out most of the data, this helper rasterises the entire swath and
    then chops it into square tiles from top (north) to bottom.  Each tile is
    resampled to ``tile_px``\*``tile_px`` pixels so it can be analysed
    independently as an image.  ``grid_res`` is automatically scaled when the
    generated grid would exceed ``max_dim`` in either dimension so extremely
    long swaths do not produce huge intermediate arrays.
    """

    lats = []
    lons = []
    rh100s = []

    for key in f.keys():
        if not key.startswith("BEAM"):
            continue
        try:
            lat = f[key]["geolocation"]["latitude"][()]
            lon = f[key]["geolocation"]["longitude"][()]
            rh = f[key]["rh_a"][()]
            if rh.ndim > 1 and rh.shape[1] > 100:
                rh100 = rh[:, 100]
            else:
                continue
            mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(rh100)
            if mask.any():
                lats.append(lat[mask])
                lons.append(lon[mask])
                rh100s.append(rh100[mask])
        except Exception:
            continue

    if not lats:
        raise ValueError("No GEDI beam data found")

    lat = np.concatenate(lats)
    lon = np.concatenate(lons)
    rh100 = np.concatenate(rh100s)

    lon_range = lon.max() - lon.min()
    lat_range = lat.max() - lat.min()
    width = int(np.ceil(lon_range / grid_res)) + 1
    height = int(np.ceil(lat_range / grid_res)) + 1
    if max(width, height) > max_dim:
        factor = max(width, height) / max_dim
        grid_res *= factor
        width = int(np.ceil(lon_range / grid_res)) + 1
        height = int(np.ceil(lat_range / grid_res)) + 1

    lon_lin = np.arange(lon.min(), lon.max() + grid_res, grid_res)
    lat_lin = np.arange(lat.max(), lat.min() - grid_res, -grid_res)
    lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

    grid = griddata((lon, lat), rh100, (lon_grid, lat_grid), method="nearest")
    grid = np.nan_to_num(grid, nan=0.0).astype(np.float32)
    arr = np.stack([grid, grid, grid], axis=2)

    import math
    from skimage.transform import resize

    height, width = arr.shape[:2]
    if width == 0 or height == 0:
        raise ValueError("Empty raster dimensions")

    tile_size = min(width, height)
    along_height = height >= width
    n_tiles = int(math.ceil(max(height, width) / tile_size))

    tiles: list[np.ndarray] = []
    for i in range(n_tiles):
        start = i * tile_size
        end = start + tile_size
        if along_height:
            patch = arr[start:end]
            if patch.shape[0] < tile_size:
                pad = tile_size - patch.shape[0]
                patch = np.pad(patch, ((0, pad), (0, 0), (0, 0)), mode="constant")
        else:
            patch = arr[:, start:end]
            if patch.shape[1] < tile_size:
                pad = tile_size - patch.shape[1]
                patch = np.pad(patch, ((0, 0), (0, pad), (0, 0)), mode="constant")
        patch = resize(
            patch,
            (tile_px, tile_px, 3),
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)
        tiles.append(patch)

    return tiles


def read_h5_image(path: Path) -> list[np.ndarray]:
    """Read a GEDI ``.h5`` file and return one or more three-band image arrays.

    When GEDI beam data is available the file is rasterised and chopped into
    square tiles.  If rasterisation fails, an exception is raised instead of
    falling back to the raw ``~100x80000`` style datasets.  For non-GEDI
    datasets the largest suitable dataset is selected without loading every
    candidate into memory.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path} is not a valid HDF5 file")

    # Attempt GEDI rasterisation first
    try:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                if (
                    key.startswith("BEAM")
                    and "geolocation" in f[key]
                    and "rh_a" in f[key]
                ):
                    try:
                        logging.info(
                            "Rasterizing GEDI tile %s via %s", path.name, key
                        )
                        tiles = _rasterize_gedi_tiles(f)
                        if tiles:
                            return tiles
                    except Exception as exc:
                        logging.warning(
                            "Rasterization failed for %s via %s: %s",
                            path.name,
                            key,
                            exc,
                        )
                        continue

            # When GEDI beams exist but rasterisation fails, avoid falling back
            # to the raw datasets which produce ~100x80000 arrays.
            for key in f.keys():
                if key.startswith("BEAM") and "rh_a" in f[key]:
                    raise ValueError("Failed to rasterize GEDI beams")

            target_name = None
            target_score = -1
            backup_name = None
            backup_size = 0
            single_channel_name = None

            def visitor(name, obj):
                nonlocal target_name, target_score, backup_name, backup_size, single_channel_name
                if not isinstance(obj, h5py.Dataset) or obj.ndim < 2:
                    return

                # Preferred: a 2-D array with many rows and a small band count
                if (
                    obj.ndim == 2
                    and obj.shape[0] > 1000
                    and obj.shape[1] in (3, 6, 8)
                ):
                    rows = obj.shape[0]
                    if rows > target_score:
                        target_name = name
                        target_score = rows

                # Backup candidate when only one band exists
                if obj.ndim == 2 and (1 in obj.shape) and single_channel_name is None:
                    single_channel_name = name

                # Fallback: keep track of the largest dataset in case no match
                size = np.prod(obj.shape)
                if size > backup_size:
                    backup_name = name
                    backup_size = size

            f.visititems(visitor)
            chosen = target_name or single_channel_name or backup_name
            if chosen is None:
                raise ValueError("No suitable dataset found")
            arr = f[chosen][...]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            logging.info(
                "Using dataset '%s' from %s with shape %s",
                chosen,
                path.name,
                arr.shape,
            )
    except OSError as exc:
        raise ValueError(f"Failed to open {path}: {exc}") from exc
    if arr.ndim == 2:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=2)
    elif arr.ndim > 3:
        logging.debug("Reshaping dataset from %s", arr.shape)
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        logging.debug("Reshaped to %s", arr.shape)
        if arr.ndim != 3 or not (1 <= arr.shape[2] <= 10):
            raise ValueError(f"Unexpected dataset shape {arr.shape}")
    if arr.shape[2] < 3:
        arr = np.repeat(arr, 3, axis=2)
    return [arr[:, :, :3]]


def to_false_color(arr: np.ndarray) -> Image.Image:
    """Convert an array to a high‑contrast false‑color ``PIL.Image``."""

    # Replace NaNs and normalise each band separately
    for i in range(arr.shape[2]):
        band = arr[..., i]
        mask = ~np.isnan(band)
        if mask.any():
            band[~mask] = 0
            band_min = band[mask].min()
            band_max = band[mask].max()
            scale = band_max - band_min
            if scale == 0:
                scale = 1.0
            band[mask] = (band[mask] - band_min) / scale
        else:
            band[:] = 0
        arr[..., i] = band

    # Stretch to 8‑bit and enhance contrast per channel
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    for i in range(3):
        if arr[..., i].max() > arr[..., i].min():
            arr[..., i] = (
                exposure.equalize_adapthist(arr[..., i]) * 255
            ).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


def query_openai(image_path: Path, tile_id: str) -> dict:
    """Send image to the OpenAI API and return structured analysis."""
    prompt = (
        "You are analyzing a Lidar-derived false color image generated from GEDI data."
        " The image has heavy foliage, vegetation, and considerable noise."
        " Search for any and all features that might reveal archaeological sites."
        " Respond in JSON with keys 'pros', 'cons', and 'confidence'."
        " 'pros' and 'cons' must each be one single line with points separated by semicolons."
        " 'confidence' must be an integer score from 1-10."
    )

    if openai.api_key:
        try:
            logging.info("Sending image %s to OpenAI", image_path)
            global openai_client
            if openai_client is None:
                openai_client = openai.OpenAI(api_key=openai.api_key)
            with image_path.open("rb") as image_file:
                b64_data = base64.b64encode(image_file.read()).decode("utf-8")
                data_url = f"data:image/png;base64,{b64_data}"

            response = openai_client.responses.create(
                model="o3",
                reasoning={"effort": "high"},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
            )
            content = response.output_text
        except Exception as exc:
            logging.error("OpenAI API call failed: %s", exc)
            content = "{}"
    else:
        # Placeholder response when no API key is configured
        logging.info("OpenAI API key missing - using placeholder response")
        content = json.dumps({
            "pros": "clearings visible; geometric shapes; elevation changes",
            "cons": "heavy vegetation; data noise; no obvious structures",
            "confidence": 5,
        })

    try:
        result = json.loads(content)
        logging.debug("OpenAI response parsed: %s", result)
        return result
    except json.JSONDecodeError as exc:
        logging.error("Failed to parse OpenAI response: %s", exc)
        return {"pros": "", "cons": "", "confidence": 0}


def process_file(
    h5_path: Path, tile_id: str, processed_dir: Path
) -> tuple[list[list], dict]:
    """Convert a single ``.h5`` file to PNG tiles and return analysis results.

    The raw ``.h5`` tile is deleted only after the image has been analysed.
    Only results with a confidence score of ``6`` or higher are kept;
    lower-confidence images are removed.  The function returns ``(rows, stats)``
    where ``rows`` is a list of CSV rows and ``stats`` tracks the counts of
    ``passed``, ``skipped`` and ``failed`` tiles.
    """

    logging.info("Processing %s", h5_path)

    rows: list[list] = []
    stats = {"passed": 0, "skipped": 0, "failed": 0}

    try:
        arrays = read_h5_image(h5_path)
        for idx, arr in enumerate(arrays, 1):
            tile_name = f"{tile_id}_{idx:03d}" if len(arrays) > 1 else tile_id

            img = to_false_color(arr)

            img_name = f"{tile_name}.png"
            img_path = processed_dir / img_name
            img.save(img_path)

            analysis = query_openai(img_path, tile_name)
            confidence = safe_cast(analysis.get("confidence", 0), int, 0)

            if confidence >= 6:
                rows.append(
                    [tile_name, analysis.get("pros", ""), analysis.get("cons", ""), confidence]
                )
                stats["passed"] += 1
            else:
                try:
                    img_path.unlink()
                    logging.info("Removed low confidence image %s", img_path)
                except Exception as exc:  # pragma: no cover - deletion best effort
                    logging.warning("Failed to delete %s: %s", img_path, exc)
                stats["skipped"] += 1

        try:
            h5_path.unlink()
            logging.info("Deleted %s", h5_path)
        except Exception as exc:  # pragma: no cover - deletion best effort
            logging.warning("Failed to delete %s: %s", h5_path, exc)

    except Exception as exc:
        logging.exception("Failed to process %s: %s", h5_path, exc)
        stats = {"passed": 0, "skipped": 0, "failed": 1}

    return (rows, stats)


def main() -> None:
    """Walk the ``data/raw`` directories and process each GEDI tile."""

    # Load API key if available to enable real OpenAI queries
    load_openai_key()

    raw_root = Path(__file__).parent / "data" / "raw"
    processed_root = Path(__file__).parent / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    name_map = {
        "riojutai": ("jutai", "jutai.csv"),
        "serra": ("serra", "serra.csv"),
        "yanomami": ("yanomami", "yanomami.csv"),
    }

    # Walk the raw data directories and process each ``.h5`` file
    for folder in raw_root.iterdir():
        if not folder.is_dir():
            continue

        proc_name, csv_name = name_map.get(folder.name, (folder.name, f"{folder.name}.csv"))
        processed_dir = processed_root / proc_name
        processed_dir.mkdir(parents=True, exist_ok=True)

        results_path = results_dir / csv_name
        existing_ids = set()
        write_header = not results_path.exists() or results_path.stat().st_size == 0

        if results_path.exists():
            with results_path.open('r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_ids.add(row.get("id", ""))

        with results_path.open('a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["id", "pros", "cons", "confidence"])

            futures = []
            max_workers = min(16, os.cpu_count() or 4)
            lock = Lock()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for h5_file in folder.glob("*.h5"):
                    try:
                        tile_id = find_tile_id(h5_file.name)
                    except ValueError:
                        logging.warning("Skipping malformed filename %s", h5_file)
                        continue
                    futures.append(
                        executor.submit(process_file, h5_file, tile_id, processed_dir)
                    )

                stats = {"total": 0, "passed": 0, "skipped": 0, "failed": 0}
                for future in as_completed(futures):
                    rows, st = future.result()
                    stats["total"] += sum(st.values())
                    for key, val in st.items():
                        stats[key] += val
                    with lock:
                        for row in rows:
                            if row[0] in existing_ids:
                                logging.info("Skipping duplicate tile %s", row[0])
                                continue
                            writer.writerow(row)
                            csvfile.flush()
                            existing_ids.add(row[0])
                logging.info(
                    "Summary for %s: total=%d passed=%d skipped=%d failed=%d",
                    folder.name,
                    stats["total"],
                    stats["passed"],
                    stats["skipped"],
                    stats["failed"],
                )


if __name__ == '__main__':
    # Entry point when executed as a script
    main()
