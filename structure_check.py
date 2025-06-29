#!/usr/bin/env python3
"""Print the internal structure of a GEDI ``.h5`` file.

The script recursively walks all groups and datasets in the provided
file and prints their names, shapes and dtypes in a readable format.
This behaves similarly to ``h5dump`` but outputs a concise Python-style
representation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py


def _print_structure(obj: h5py.Group | h5py.Dataset, name: str, indent: int = 0) -> None:
    """Recursively print the structure of ``obj``.

    Parameters
    ----------
    obj:
        The current ``h5py`` object (group or dataset).
    name:
        Full path name of ``obj`` within the file.
    indent:
        Current indentation level for pretty printing.
    """
    pad = " " * indent
    if isinstance(obj, h5py.Dataset):
        print(f"{pad}{name} \u2013 dataset shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{pad}{name}/")
        for key in obj:
            item = obj[key]
            sub_name = f"{name}/{key}" if name != "/" else f"/{key}"
            _print_structure(item, sub_name, indent + 2)


def main() -> None:
    """Parse arguments and print the structure of the given ``.h5`` file."""
    parser = argparse.ArgumentParser(description="Display the structure of a GEDI .h5 file")
    parser.add_argument("file", type=Path, help="Path to the .h5 file")
    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"{args.file} does not exist")
    if not h5py.is_hdf5(args.file):
        parser.error(f"{args.file} is not a valid HDF5 file")

    with h5py.File(args.file, "r") as f:
        _print_structure(f, "/")


if __name__ == "__main__":
    main()
