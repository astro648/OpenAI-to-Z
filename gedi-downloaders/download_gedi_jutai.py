"""Download GEDI granules for the Rio Jutai headwaters."""

import utils

BBOX = (-68.5, -6.0, -68.5, -4.5)  # Rio Jutai Headwaters
TARGET = utils.data_path("riojutai")


LIMIT = 200


def main() -> None:
    utils.login()
    granules = utils.search("GEDI02_A", BBOX)
    print(f"Found {len(granules)} GEDI granules in Rio Jutai headwaters")
    utils.download(granules, TARGET, limit=LIMIT)


if __name__ == "__main__":
    main()

