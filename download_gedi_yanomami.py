from earthaccess import Auth, DataGranules, Download

auth = Auth().login(strategy="browser")
print("Authenticated:", auth.authenticated)

granules = DataGranules().search(
    short_name="GEDI02_A",
    bounding_box=(-65.0, 1.0, -62.0, 4.0),  # Yanomami Frontier
    cloud_hosted=True,
)

print(f"Found {len(granules)} GEDI granules in Yanomami region")
Download().download(granules, "./data/raw/yanomami/")
