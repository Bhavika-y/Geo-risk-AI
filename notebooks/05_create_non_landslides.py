import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# ============================================================
# 1. LOAD THE CLEANED LANDSLIDE DATA
# ============================================================

file_path = "data/processed/sikkim_landslides_cleaned.geojson"

gdf = gpd.read_file(file_path)

print("Known landslides:", len(gdf))


# ============================================================
# 2. CREATE A STUDY-AREA BOUNDARY
# ============================================================

study_area = gdf.geometry.union_all().convex_hull

print("\nStudy area boundary created.")


# ============================================================
# 3. GENERATE RANDOM POINTS
# ============================================================

min_x, min_y, max_x, max_y = study_area.bounds

np.random.seed(42)

random_points = []

target_points = len(gdf)

while len(random_points) < target_points:

    x = np.random.uniform(min_x, max_x)
    y = np.random.uniform(min_y, max_y)

    point = Point(x, y)

    if study_area.contains(point):
        random_points.append(point)


# ============================================================
# 4. CONVERT RANDOM POINTS INTO A GEODATAFRAME
# ============================================================

non_landslides = gpd.GeoDataFrame(
    geometry=random_points,
    crs=gdf.crs
)

print("Random points generated:", len(non_landslides))


# ============================================================
# 5. LABEL THEM AS NON-LANDSLIDES
# ============================================================

non_landslides["landslide"] = 0


# ============================================================
# 6. LABEL THE ORIGINAL LANDSLIDES
# ============================================================

gdf["landslide"] = 1


# ============================================================
# 7. COMBINE BOTH DATASETS
# ============================================================

combined = gpd.GeoDataFrame(
    pd.concat([gdf, non_landslides], ignore_index=True),
    crs=gdf.crs
)

print("\n===== FINAL SAMPLE DATA =====")
print(combined["landslide"].value_counts())


# ============================================================
# 8. SAVE THE DATASET
# ============================================================

output_file = "data/processed/sikkim_landslide_binary_samples.geojson"

combined.to_file(output_file, driver="GeoJSON")

print("\nDataset saved to:")
print(output_file)