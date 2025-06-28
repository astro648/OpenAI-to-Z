import earthaccess

earthaccess.login(strategy="browser")

granules = earthaccess.search_data(
    short_name="GEDI02_A",
    bounding_box=(-68.5, -6.0, -68.5, -4.5),  # Rio Jutai Headwaters
    cloud_hosted=True
)

print(f"Found {len(granules)} GEDI granules in Rio Jutai Headwaters")

earthaccess.download(granules, "./data/raw/jutai/")
