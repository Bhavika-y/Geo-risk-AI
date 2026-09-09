import geopandas as gpd
import rasterio
import numpy as np


# ==============================
# 1. FILE PATHS
# ==============================

points_file = "data/processed/sikkim_landslide_binary_samples_v2.geojson"
rainfall_file = r"C:\Users\HP\Downloads\chirps-v2.0.2019.tif"
output_file = "data/processed/sikkim_samples_with_rainfall.geojson"


# ==============================
# 2. LOAD LANDSLIDE DATA
# ==============================

gdf = gpd.read_file(points_file)

print("Number of samples:", len(gdf))
print("Sample CRS:", gdf.crs)


# ==============================
# 3. OPEN RAINFALL DATA
# ==============================

with rasterio.open(rainfall_file) as src:

    print("\n===== RAINFALL DATA =====")
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)

    rainfall = src.read(1)

    print(
        "Minimum rainfall:",
        np.nanmin(rainfall)
    )

    print(
        "Maximum rainfall:",
        np.nanmax(rainfall)
    )

    # Convert points to rainfall CRS
    gdf_rain = gdf.to_crs(src.crs)

    # Extract rainfall values
    coordinates = [
        (point.x, point.y)
        for point in gdf_rain.geometry
    ]

    rainfall_values = []

    for value in src.sample(coordinates):

        rainfall_values.append(value[0])


# ==============================
# 4. ADD RAINFALL FEATURE
# ==============================

gdf_rain["rainfall_2019"] = rainfall_values


# ==============================
# 5. CHECK RESULTS
# ==============================

print("\n===== RAINFALL FEATURE =====")

print(
    gdf_rain["rainfall_2019"].describe()
)

print("\nMissing rainfall values:")

print(
    gdf_rain["rainfall_2019"].isna().sum()
)


# ==============================
# 6. SAVE DATASET
# ==============================

gdf_rain = gdf_rain.to_crs("EPSG:4326")

gdf_rain.to_file(
    output_file,
    driver="GeoJSON"
)

print("\nDataset saved to:")
print(output_file)