import earthaccess
from pathlib import Path

# Load Earthdata credentials from the repository root
ROOT = Path(__file__).resolve().parent.parent
username_path = ROOT / "username.txt"
password_path = ROOT / "password.txt"
username = username_path.read_text().strip() if username_path.exists() else ""
password = password_path.read_text().strip() if password_path.exists() else ""

# Authenticate using the stored credentials
# ``earthaccess.login`` expects positional arguments for the username and
# password rather than keyword arguments.
earthaccess.login(username, password)

granules = earthaccess.search_data(
    short_name="GEDI02_A",
    bounding_box=(-65.0, 1.0, -62.0, 4.0),  # Yanomami Frontier
    cloud_hosted=True
)

print(f"Found {len(granules)} GEDI granules in Yanomami Frontier")

earthaccess.download(granules, "./data/raw/yanomami/")
