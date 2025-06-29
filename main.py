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
from io import BytesIO
import argparse

import h5py
import numpy as np
from PIL import Image
from skimage import exposure
from scipy.interpolate import griddata
from numpy.typing import ArrayLike
import openai
import csv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Path to API key file located at repository root
API_KEY_FILE = Path(__file__).resolve().parent / "openai_api_key.txt"


def configure_compute_threads(workers: int) -> None:
    """Limit numpy/BLAS thread usage when using multiple workers."""
    if workers > 1:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def load_openai_client(timeout: float = 60.0) -> typing.Optional[openai.OpenAI]:
    """Load the OpenAI API key and return a client instance if available.

    A ``timeout`` can be provided to prevent API requests from hanging
    indefinitely.  The value is passed directly to :class:`openai.OpenAI`.
    """
    try:
        key = API_KEY_FILE.read_text().strip()
        if key:
            logging.info(
                "Loaded OpenAI API key from %s with timeout=%ss",
                API_KEY_FILE,
                timeout,
            )
            return openai.OpenAI(api_key=key, timeout=timeout)
        logging.warning("OpenAI API key file %s is empty", API_KEY_FILE)
    except FileNotFoundError:
        logging.warning("OpenAI API key file %s not found", API_KEY_FILE)
    return None

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


def _grid_average(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
) -> np.ndarray:
    """Return the average of ``values`` aggregated into the specified grid."""
    sum_grid, _, _ = np.histogram2d(
        lat, lon, bins=[lat_edges, lon_edges], weights=values
    )
    count_grid, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.divide(
            sum_grid,
            count_grid,
            out=np.zeros_like(sum_grid, dtype=np.float32),
            where=count_grid > 0,
        )
    return avg.astype(np.float32)


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

    logging.info(
        "Starting rasterization grid_res=%s tile_px=%s max_dim=%s",
        grid_res,
        tile_px,
        max_dim,
    )

    lats = []
    lons = []
    rh100s = []
    elevs = []
    energies = []

    for key in f.keys():
        if not key.startswith("BEAM"):
            continue
        logging.info("Processing %s", key)
        beam = f[key]
        if "geolocation" not in beam:
            logging.info("%s missing geolocation", key)
            continue
        geo = beam["geolocation"]
        for algo in range(1, 7):
            logging.info("  Algorithm %s", algo)
            lat_name = f"lat_highestreturn_a{algo}"
            lon_name = f"lon_highestreturn_a{algo}"
            rh_name = f"rh_a{algo}"
            elev_name = f"elev_highestreturn_a{algo}"
            qual_name = f"quality_flag_a{algo}"
            if lat_name not in geo or lon_name not in geo or rh_name not in geo:
                logging.info("    Required datasets missing for a%s", algo)
                continue
            try:
                lat = geo[lat_name][()]
                lon = geo[lon_name][()]
                rh = geo[rh_name][()]
                elev = None
                if elev_name in geo:
                    elev = geo[elev_name][()]
                elif "elevation_1gfit" in geo:
                    elev = geo["elevation_1gfit"][()]
                if rh.ndim > 1 and rh.shape[1] > 100:
                    rh100 = rh[:, 100].astype(np.float32) / 100.0
                else:
                    continue
                rx_group = beam.get(f"rx_processing_a{algo}")
                energy = None
                if isinstance(rx_group, h5py.Group):
                    if "rx_maxamp" in rx_group:
                        energy = rx_group["rx_maxamp"][()]
                    elif "rx_energy" in rx_group:
                        energy = rx_group["rx_energy"][()]
                quality_mask = np.ones_like(rh100, dtype=bool)
                if qual_name in geo:
                    quality_mask &= geo[qual_name][()] == 1
                if "degrade_flag" in geo:
                    quality_mask &= geo["degrade_flag"][()] == 0
                if "stale_return_flag" in geo:
                    quality_mask &= geo["stale_return_flag"][()] == 0
                mask = (
                    np.isfinite(lat)
                    & np.isfinite(lon)
                    & np.isfinite(rh100)
                    & (rh100 > 0)
                    & quality_mask
                )
                if elev is not None:
                    mask &= np.isfinite(elev)
                if energy is not None:
                    mask &= np.isfinite(energy)
                if mask.any():
                    lats.append(lat[mask])
                    lons.append(lon[mask])
                    rh100s.append(rh100[mask])
                    if elev is not None:
                        elevs.append(elev[mask])
                    if energy is not None:
                        energies.append(energy[mask])
                    logging.info(
                        "    Collected %s points from %s algo %s",
                        mask.sum(),
                        key,
                        algo,
                    )
            except Exception:
                continue

    if not lats:
        raise ValueError("No GEDI beam data found")

    lat = np.concatenate(lats)
    lon = np.concatenate(lons)
    rh100 = np.concatenate(rh100s)
    elev = np.concatenate(elevs) if elevs else np.zeros_like(rh100)
    energy = np.concatenate(energies) if energies else np.zeros_like(rh100)
    logging.info("Total points after merge: %s", lat.size)

    lat_q1, lat_q3 = np.percentile(lat, [25, 75])
    lon_q1, lon_q3 = np.percentile(lon, [25, 75])
    lat_iqr = lat_q3 - lat_q1
    lon_iqr = lon_q3 - lon_q1
    lat_low = lat_q1 - 1.5 * lat_iqr
    lat_high = lat_q3 + 1.5 * lat_iqr
    lon_low = lon_q1 - 1.5 * lon_iqr
    lon_high = lon_q3 + 1.5 * lon_iqr
    keep = (
        (lat >= lat_low)
        & (lat <= lat_high)
        & (lon >= lon_low)
        & (lon <= lon_high)
    )
    removed = lat.size - keep.sum()
    lat = lat[keep]
    lon = lon[keep]
    rh100 = rh100[keep]
    elev = elev[keep]
    energy = energy[keep]
    logging.info("Removed %s outliers", removed)

    lon_range = lon.max() - lon.min()
    lat_range = lat.max() - lat.min()
    width = int(np.ceil(lon_range / grid_res)) + 1
    height = int(np.ceil(lat_range / grid_res)) + 1
    if max(width, height) > max_dim:
        factor = max(width, height) / max_dim
        grid_res *= factor
        width = int(np.ceil(lon_range / grid_res)) + 1
        height = int(np.ceil(lat_range / grid_res)) + 1

    lon_edges = np.arange(lon.min(), lon.max() + grid_res, grid_res)
    lat_edges = np.arange(lat.min(), lat.max() + grid_res, grid_res)
    logging.info(
        "Raster grid %sx%s with resolution %s",
        len(lat_edges) - 1,
        len(lon_edges) - 1,
        grid_res,
    )

    grid_rh = _grid_average(lon, lat, rh100, lon_edges, lat_edges)[::-1]
    grid_elev = _grid_average(lon, lat, elev, lon_edges, lat_edges)[::-1]
    grid_energy = _grid_average(lon, lat, energy, lon_edges, lat_edges)[::-1]

    arr = np.stack([grid_rh, grid_elev, grid_energy], axis=2)
    logging.info("Stacked raster array shape %s", arr.shape)

    import math
    from skimage.transform import resize

    height, width = arr.shape[:2]
    if width == 0 or height == 0:
        raise ValueError("Empty raster dimensions")

    tile_size = min(width, height)
    along_height = height >= width
    n_tiles = int(math.ceil(max(height, width) / tile_size))
    logging.info("Creating %s tiles", n_tiles)

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
        logging.info("  Rasterizing tile %s/%s", i + 1, n_tiles)
        patch = resize(
            patch,
            (tile_px, tile_px, 3),
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)
        tiles.append(patch)
    logging.info("Rasterization completed with %s tiles", len(tiles))

    return tiles


def read_h5_image(path: Path) -> list[np.ndarray]:
    """Read and rasterise a GEDI ``.h5`` file into square image tiles.

    All input files must contain GEDI beam data.  The beams are combined,
    rasterised and chopped into equally sized tiles which are returned as a
    list of three-band arrays.  Failure to produce these tiles results in an
    exception and the file is considered invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path} is not a valid HDF5 file")
    logging.info("Opening %s", path)

    try:
        with h5py.File(path, "r") as f:
            logging.info("Scanning beams")
            has_beam = False
            for key in f.keys():
                if not key.startswith("BEAM"):
                    continue
                logging.info("Found %s", key)
                beam = f[key]
                geo = beam.get("geolocation")
                if not isinstance(geo, h5py.Group):
                    logging.info("%s missing geolocation", key)
                    continue
                for algo in range(1, 7):
                    if (
                        f"lat_highestreturn_a{algo}" in geo
                        and f"lon_highestreturn_a{algo}" in geo
                        and f"rh_a{algo}" in geo
                    ):
                        logging.info("Beam %s supports algorithm %s", key, algo)
                        has_beam = True
                        break
                if has_beam:
                    break
            if not has_beam:
                logging.info("No GEDI beam data found in %s", path.name)
                raise ValueError(f"No GEDI beam data found in {path.name}")

            logging.info("Rasterizing GEDI tile %s", path.name)
            tiles = _rasterize_gedi_tiles(f)
            if not tiles:
                raise ValueError("Rasterization produced no tiles")
            logging.info("Rasterization yielded %s tiles", len(tiles))
            return tiles
    except OSError as exc:
        raise ValueError(f"Failed to open {path}: {exc}") from exc


def to_false_color(arr: np.ndarray) -> Image.Image:
    """Convert a 3-band array to a high‑contrast false‑color ``PIL.Image``.

    The input array should contain canopy height, terrain elevation and
    waveform energy in separate channels.  Each band is normalised and
    contrast-enhanced independently before merging into an RGB image.
    """

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


def query_openai(img: Image.Image, tile_id: str, client: typing.Optional[openai.OpenAI]) -> dict:
    """Send image to the OpenAI API and return structured analysis.

    The function **requires** a valid ``OpenAI`` client.  If ``client`` is ``None``
    or the API request fails, a ``RuntimeError`` is raised so the caller can
    terminate processing without writing placeholder data.
    """
    prompt = (
        "You are analyzing a Lidar-derived false color image generated from GEDI data."
        " The image has heavy foliage, vegetation, and considerable noise."
        " Search for any and all features that might reveal archaeological sites."
        " Respond in JSON with keys 'pros', 'cons', and 'confidence'."
        " 'pros' and 'cons' must each be one single line with points separated by semicolons."
        " 'confidence' must be an integer score from 1-10."
    )

    if client is None:
        raise RuntimeError("OpenAI client not configured")

    try:
        logging.info("Sending tile %s to OpenAI", tile_id)
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_data}"
        response = client.responses.create(
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
        raise RuntimeError("OpenAI API request failed") from exc

    try:
        result = json.loads(content)
        logging.debug("OpenAI response parsed: %s", result)
        return result
    except json.JSONDecodeError as exc:
        logging.error("Failed to parse OpenAI response: %s", exc)
        return {"pros": "", "cons": "", "confidence": 0}


def process_file(
    h5_path: Path, tile_prefix: str, processed_dir: Path, client: typing.Optional[openai.OpenAI]
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

    arrays = read_h5_image(h5_path)
    for idx, arr in enumerate(arrays, 1):
        tile_name = f"{tile_prefix}_{idx:03d}" if len(arrays) > 1 else tile_prefix

        img = to_false_color(arr)

        analysis = query_openai(img, tile_name, client)
        confidence = safe_cast(analysis.get("confidence", 0), int, 0)

        if confidence >= 6:
            img_name = f"{tile_name}.png"
            img_path = processed_dir / img_name
            img.save(img_path)
            rows.append(
                [tile_name, analysis.get("pros", ""), analysis.get("cons", ""), confidence]
            )
            stats["passed"] += 1
        else:
            stats["skipped"] += 1

    h5_path.unlink(missing_ok=True)
    logging.info("Deleted %s", h5_path)


    return (rows, stats)


def main() -> None:
    """Walk the ``data/raw`` directories and process each GEDI tile."""

    parser = argparse.ArgumentParser(description="Process GEDI tiles")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(__file__).parent / "data" / "raw",
        help="Directory containing raw .h5 files",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(__file__).parent / "data" / "processed",
        help="Directory to write processed images",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory to store result CSV files",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="OpenAI request timeout in seconds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="Number of worker threads for tile processing",
    )
    args = parser.parse_args()

    configure_compute_threads(args.workers)

    openai_client = load_openai_client(timeout=args.timeout)
    if openai_client is None:
        logging.error("OpenAI API key missing - aborting")
        return

    raw_root = args.raw_root
    processed_root = args.processed_root
    processed_root.mkdir(parents=True, exist_ok=True)

    results_dir = args.results_dir
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
            max_workers = max(1, args.workers)
            lock = Lock()
            logging.info("Starting thread pool with %s workers", max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for h5_file in folder.glob("*.h5"):
                    try:
                        tile_id = find_tile_id(h5_file.name)
                    except ValueError:
                        logging.warning("Skipping malformed filename %s", h5_file)
                        continue
                    tile_prefix = f"{folder.name}_{tile_id}"
                    futures.append(
                        executor.submit(process_file, h5_file, tile_prefix, processed_dir, openai_client)
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
                            existing_ids.add(row[0])
                csvfile.flush()
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
