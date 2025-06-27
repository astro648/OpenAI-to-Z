from earthaccess import Auth, DataGranules, Download

auth = Auth().login(strategy="browser")
print("Authenticated:", auth.authenticated)

granules = DataGranules().search(
    short_name="GEDI02_A",
    bounding_box=(-74.0, -8.0, -72.5, -6.0),  # Serra do Divisor
    cloud_hosted=True,
)

print(f"Found {len(granules)} GEDI granules in Serra do Divisor")
Download().download(granules, "./data/raw/serra/")
