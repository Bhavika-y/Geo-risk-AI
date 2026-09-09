import geopandas as gpd
import pandas as pd

# ============================================================
# 1. LOAD THE LANDSLIDE INVENTORY
# ============================================================

file_path = "data/raw/sikkim_landslides/Google_Earth_landslides_point_21Dec2021.shp"

gdf = gpd.read_file(file_path)

print("Original number of records:", len(gdf))


# ============================================================
# 2. KEEP ONLY THE FEATURES WE NEED
# ============================================================

columns_to_keep = [
    "Name",
    "Aspect",
    "Curvature",
    "Slope",
    "Elevation",
    "Geology",
    "Extent",
    "geometry"
]

gdf = gdf[columns_to_keep].copy()


# ============================================================
# 3. REMOVE RECORDS WITH MISSING IMPORTANT VALUES
# ============================================================

important_columns = [
    "Aspect",
    "Curvature",
    "Slope",
    "Elevation",
    "Geology",
    "geometry"
]

before_cleaning = len(gdf)

gdf = gdf.dropna(subset=important_columns)

after_cleaning = len(gdf)

print("\nRecords before cleaning:", before_cleaning)
print("Records after cleaning:", after_cleaning)
print("Records removed:", before_cleaning - after_cleaning)


# ============================================================
# 4. CHECK FOR DUPLICATE RECORDS
# ============================================================

duplicates = gdf.duplicated().sum()

print("\nDuplicate records:", duplicates)

gdf = gdf.drop_duplicates()


# ============================================================
# 5. CREATE LANDSLIDE TARGET LABEL
# ============================================================

gdf["landslide"] = 1


# ============================================================
# 6. CHECK THE CLEAN DATA
# ============================================================

print("\n===== CLEAN DATASET =====")

print("Number of records:", len(gdf))

print("\nColumns:")
print(gdf.columns.tolist())

print("\nMissing values:")
print(gdf.isnull().sum())

print("\nTarget distribution:")
print(gdf["landslide"].value_counts())


# ============================================================
# 7. SAVE THE CLEAN DATASET
# ============================================================

output_file = "data/processed/sikkim_landslides_cleaned.geojson"

gdf.to_file(output_file, driver="GeoJSON")

print("\nClean dataset saved to:")
print(output_file)