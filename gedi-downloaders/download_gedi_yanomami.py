"""Download GEDI granules for the Yanomami frontier."""

import utils

BBOX = (-65.0, 1.0, -62.0, 4.0)  # Yanomami Frontier
TARGET = utils.data_path("yanomami")


LIMIT = 200


def main() -> None:
    utils.login()
    granules = utils.search("GEDI02_A", BBOX)
    print(f"Found {len(granules)} GEDI granules in Yanomami frontier")
    utils.download(granules, TARGET, limit=LIMIT)


if __name__ == "__main__":
    main()

