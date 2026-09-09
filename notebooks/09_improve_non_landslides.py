import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point


# ==============================
# 1. LOAD LANDSLIDE DATA
# ==============================

landslide_file = "data/processed/sikkim_landslides_cleaned.geojson"

gdf = gpd.read_file(landslide_file)

print("Known landslides:", len(gdf))
print("Original CRS:", gdf.crs)


# ==============================
# 2. CONVERT TO METERS
# ==============================

target_crs = "EPSG:32645"

gdf_utm = gdf.to_crs(target_crs)

print("Converted to:", target_crs)


# ==============================
# 3. CREATE STUDY AREA
# ==============================

study_area = gdf_utm.geometry.union_all().convex_hull

print("Study area created.")


# ==============================
# 4. CREATE EXCLUSION ZONE
# ==============================

# Keep random non-landslide points
# at least 500 meters away from
# known landslide locations.

exclusion_distance = 500

landslide_buffer = gdf_utm.geometry.buffer(
    exclusion_distance
)

exclusion_zone = landslide_buffer.union_all()

print(
    "Exclusion distance:",
    exclusion_distance,
    "meters"
)


# ==============================
# 5. GENERATE RANDOM POINTS
# ==============================

min_x, min_y, max_x, max_y = study_area.bounds

np.random.seed(42)

random_points = []

target_points = len(gdf_utm)

while len(random_points) < target_points:

    x = np.random.uniform(min_x, max_x)
    y = np.random.uniform(min_y, max_y)

    point = Point(x, y)

    if (
        study_area.contains(point)
        and not exclusion_zone.contains(point)
    ):
        random_points.append(point)


print(
    "Valid non-landslide points:",
    len(random_points)
)


# ==============================
# 6. CREATE NON-LANDSLIDE DATA
# ==============================

non_landslides = gpd.GeoDataFrame(
    geometry=random_points,
    crs=target_crs
)

non_landslides["landslide"] = 0

gdf_utm["landslide"] = 1


# ==============================
# 7. COMBINE DATA
# ==============================

combined = gpd.GeoDataFrame(
    pd.concat(
        [gdf_utm, non_landslides],
        ignore_index=True
    ),
    crs=target_crs
)


# ==============================
# 8. CHECK CLASS BALANCE
# ==============================

print("\n===== CLASS DISTRIBUTION =====")

print(
    combined["landslide"].value_counts()
)


# ==============================
# 9. CONVERT BACK TO WGS84
# ==============================

combined = combined.to_crs("EPSG:4326")


# ==============================
# 10. SAVE DATASET
# ==============================

output_file = (
    "data/processed/"
    "sikkim_landslide_binary_samples_v2.geojson"
)

combined.to_file(
    output_file,
    driver="GeoJSON"
)

print("\nDataset saved to:")
print(output_file)