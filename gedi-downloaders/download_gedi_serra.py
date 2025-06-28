"""Download GEDI granules for the Serra do Divisor region."""

import utils

BBOX = (-74.0, -8.0, -72.5, -6.0)  # Serra do Divisor
TARGET = utils.data_path("serra")


def main() -> None:
    utils.login()
    granules = utils.search("GEDI02_A", BBOX)
    print(f"Found {len(granules)} GEDI granules in Serra do Divisor")
    utils.download(granules, TARGET)


if __name__ == "__main__":
    main()

