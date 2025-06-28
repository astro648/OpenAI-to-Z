import earthaccess
from pathlib import Path

# Load Earthdata credentials from the repository root
ROOT = Path(__file__).resolve().parent.parent
username_path = ROOT / "username.txt"
password_path = ROOT / "password.txt"
username = username_path.read_text().strip() if username_path.exists() else ""
password = password_path.read_text().strip() if password_path.exists() else ""

# Authenticate using the stored credentials. Newer versions of
# ``earthaccess`` accept ``username`` and ``password`` as keyword arguments
# while older releases expect them as positional parameters.  Attempt the
# keyword form first then fall back to the positional call for maximum
# compatibility.
try:
    earthaccess.login(username=username, password=password)
except TypeError as exc:
    if "unexpected keyword argument" in str(exc):
        earthaccess.login(username, password)
    else:
        raise

granules = earthaccess.search_data(
    short_name="GEDI02_A",
    bounding_box=(-65.0, 1.0, -62.0, 4.0),  # Yanomami Frontier
    cloud_hosted=True
)

print(f"Found {len(granules)} GEDI granules in Yanomami Frontier")

earthaccess.download(granules, "./data/raw/yanomami/")
