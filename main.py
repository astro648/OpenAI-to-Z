"""Process GEDI Lidar tiles and query OpenAI for archaeological clues.

Each ``.h5`` file under ``data/raw`` is converted into a high‑contrast
false‑color PNG image saved in ``data/processed``. The image is then analysed
using the OpenAI API and the results appended to ``results.csv``. The CSV
contains the columns ``id``, ``pros``, ``cons`` and ``confidence``.

The script is intentionally verbose with logging to aid in tracing processing
and troubleshooting malformed files. The OpenAI API key is expected in a file
named ``openai_api_key.txt`` located at the repository root.
"""

import re
import json
import logging
from pathlib import Path

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
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path} is not a valid HDF5 file")

    # Determine the largest 2-D or 3-D dataset path first
    with h5py.File(path, "r") as f:
        target_name = None
        max_elems = 0

        def visitor(name, obj):
            nonlocal target_name, max_elems
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                size = np.prod(obj.shape)
                if size > max_elems:
                    target_name = name
                    max_elems = size

        f.visititems(visitor)
        if target_name is None:
            raise ValueError("No suitable dataset found")
        arr = f[target_name][...].astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    if arr.shape[2] < 3:
        arr = np.repeat(arr, 3, axis=2)
    return arr[:, :, :3]


def to_false_color(arr: np.ndarray) -> Image.Image:
    """Convert an array to a high‑contrast false‑color ``PIL.Image``."""

    # Replace NaN values and normalise to 0‑1 range
    arr = np.nan_to_num(arr)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val

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


def process_file(h5_path: Path, tile_id: str, processed_dir: Path, writer: csv.writer) -> None:
    """Convert a single ``.h5`` file to PNG and record OpenAI analysis."""

    logging.info("Processing %s", h5_path)

    try:
        arr = read_h5_image(h5_path)
        img = to_false_color(arr)

        # Save processed image using the tile id for traceability
        img_name = f"{tile_id}.png"
        img_path = processed_dir / img_name
        img.save(img_path)

        # Query the OpenAI API and append the results
        analysis = query_openai(img_path, tile_id)
        writer.writerow([
            tile_id,
            analysis.get("pros", ""),
            analysis.get("cons", ""),
            analysis.get("confidence", ""),
        ])
    except Exception as exc:
        logging.exception("Failed to process %s: %s", h5_path, exc)


def main() -> None:
    """Walk the ``data/raw`` directories and process each GEDI tile."""

    # Load API key if available to enable real OpenAI queries
    load_openai_key()

    raw_root = Path(__file__).parent / "data" / "raw"
    processed_dir = Path(__file__).parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(__file__).parent / "results.csv"
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
            # Create header on first run so the CSV is immediately usable
            writer.writerow(["id", "pros", "cons", "confidence"])

        # Walk the raw data directories and process each ``.h5`` file
        for folder in raw_root.iterdir():
            if folder.is_dir():
                for h5_file in folder.glob("*.h5"):
                    try:
                        tile_id = find_tile_id(h5_file.name)
                    except ValueError:
                        logging.warning("Skipping malformed filename %s", h5_file)
                        continue
                    if tile_id in existing_ids:
                        logging.info("Skipping %s - already processed", tile_id)
                        continue
                    process_file(h5_file, tile_id, processed_dir, writer)
                    existing_ids.add(tile_id)


if __name__ == '__main__':
    # Entry point when executed as a script
    main()
