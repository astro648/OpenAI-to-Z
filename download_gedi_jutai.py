from earthaccess import Auth, DataGranules, Download

auth = Auth().login(strategy="browser")
print("Authenticated:", auth.authenticated)

granules = DataGranules().search(
    short_name="GEDI02_A",
    bounding_box=(-68.5, -6.0, -68.5, -4.5),  # Rio Jutai Headwaters
    cloud_hosted=True,
)

print(f"Found {len(granules)} GEDI granules in Rio Jutai region")
Download().download(granules, "./data/raw/jutai/")
