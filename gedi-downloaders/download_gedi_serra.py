import earthaccess

earthaccess.login(strategy="browser")

granules = earthaccess.search_data(
    short_name="GEDI02_A",
    bounding_box=(-74.0, -8.0, -72.5, -6.0),  # Serra do Divisor
    cloud_hosted=True
)

print(f"Found {len(granules)} GEDI granules in Serra do Divisor")

earthaccess.download(granules, "./data/raw/serra/")
