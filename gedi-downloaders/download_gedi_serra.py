import earthaccess
from pathlib import Path

# Load Earthdata credentials from the repository root
ROOT = Path(__file__).resolve().parent.parent
username_path = ROOT / "username.txt"
password_path = ROOT / "password.txt"
username = username_path.read_text().strip() if username_path.exists() else ""
password = password_path.read_text().strip() if password_path.exists() else ""

# Authenticate using the stored credentials. ``earthaccess.login`` expects the
# username and password to be provided as keyword arguments; passing them as
# positional values can cause the library to treat the username as the login
# strategy and fail to initialise the session.
earthaccess.login(username=username, password=password)

granules = earthaccess.search_data(
    short_name="GEDI02_A",
    bounding_box=(-74.0, -8.0, -72.5, -6.0),  # Serra do Divisor
    cloud_hosted=True
)

print(f"Found {len(granules)} GEDI granules in Serra do Divisor")

earthaccess.download(granules, "./data/raw/serra/")
