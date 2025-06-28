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
from pathlib import Path
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
from PIL import Image
from skimage import exposure
import openai
import csv

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


def read_h5_image(path: Path) -> np.ndarray:
    """Read a GEDI ``.h5`` file and return a three-band image array.

    The file may contain multiple datasets, so the dataset with the greatest
    number of elements is selected without loading each candidate into memory.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path} is not a valid HDF5 file")

    # Choose a dataset based on heuristics rather than raw size
    try:
        with h5py.File(path, "r") as f:
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
    if arr.shape[2] < 3:
        arr = np.repeat(arr, 3, axis=2)
    return arr[:, :, :3]


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
            client = openai.OpenAI(api_key=openai.api_key)
            with image_path.open("rb") as image_file:
                response = client.responses.create(
                    model="o3",
                    reasoning={"effort": "high"},
                    input=[
                        {"role": "user", "content": prompt},
                        {"role": "user", "image": image_file},
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
) -> typing.Optional[list]:
    """Convert a single ``.h5`` file to PNG and return analysis results.

    The ``.h5`` file is removed after it has been read. Only results with a
    confidence score of ``6`` or higher are kept; lower-confidence images are
    deleted along with their corresponding images. If the confidence meets the
    threshold, a list containing the CSV row data is returned.
    """

    logging.info("Processing %s", h5_path)

    try:
        arr = read_h5_image(h5_path)
        img = to_false_color(arr)

        # Delete the original h5 tile once we have the image in memory
        try:
            h5_path.unlink()
            logging.info("Deleted %s", h5_path)
        except Exception as exc:  # pragma: no cover - deletion best effort
            logging.warning("Failed to delete %s: %s", h5_path, exc)

        # Save processed image using the tile id for traceability
        img_name = f"{tile_id}.png"
        img_path = processed_dir / img_name
        img.save(img_path)

        # Query the OpenAI API and append the results
        analysis = query_openai(img_path, tile_id)
        try:
            confidence = int(float(analysis.get("confidence", 0)))
        except (ValueError, TypeError):
            logging.warning("Invalid confidence value for tile %s", tile_id)
            confidence = 0

        if confidence >= 6:
            return [
                tile_id,
                analysis.get("pros", ""),
                analysis.get("cons", ""),
                confidence,
            ]
        else:
            # Remove low-confidence image and skip CSV entry
            try:
                img_path.unlink()
                logging.info("Removed low confidence image %s", img_path)
            except Exception as exc:  # pragma: no cover - deletion best effort
                logging.warning("Failed to delete %s: %s", img_path, exc)
    except Exception as exc:
        logging.exception("Failed to process %s: %s", h5_path, exc)
    return None


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
            with ThreadPoolExecutor(max_workers=4) as executor:
                for h5_file in folder.glob("*.h5"):
                    try:
                        tile_id = find_tile_id(h5_file.name)
                    except ValueError:
                        logging.warning("Skipping malformed filename %s", h5_file)
                        continue
                    if tile_id in existing_ids:
                        logging.info("Skipping %s - already processed", tile_id)
                        continue
                    futures.append(
                        executor.submit(process_file, h5_file, tile_id, processed_dir)
                    )
                    existing_ids.add(tile_id)

                for future in as_completed(futures):
                    row = future.result()
                    if row:
                        writer.writerow(row)
                        csvfile.flush()


if __name__ == '__main__':
    # Entry point when executed as a script
    main()
