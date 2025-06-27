import earthaccess

earthaccess.login(strategy="browser")

granules = earthaccess.search_data(
    short_name="GEDI02_A",
    bounding_box=(-65.0, 1.0, -62.0, 4.0),  # Yanomami Frontier
    cloud_hosted=True
)

print(f"Found {len(granules)} GEDI granules in Yanomami Frontier")

earthaccess.download(granules, "./data/raw/yanomami/")
