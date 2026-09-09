import geopandas as gpd

# Load the Sikkim landslide dataset
file_path = "data/raw/sikkim_landslides/Google_Earth_landslides_point_21Dec2021.shp"

gdf = gpd.read_file(file_path)


# ============================================================
# 1. BASIC DATASET INFORMATION
# ============================================================

print("===== DATASET SIZE =====")
print("Number of landslides:", len(gdf))
print("Number of columns:", len(gdf.columns))

print("\nColumn names:")
print(gdf.columns.tolist())


# ============================================================
# 2. NUMERICAL FEATURES
# ============================================================

print("\n===== NUMERICAL FEATURES =====")

for column in ["Slope", "Elevation", "Aspect", "Curvature"]:
    print(f"\n--- {column} ---")
    print(gdf[column].describe())


# ============================================================
# 3. GEOLOGY
# ============================================================

print("\n===== GEOLOGY =====")
print(gdf["Geology"].value_counts())


# ============================================================
# 4. LANDSLIDE TYPES
# ============================================================

print("\n===== LANDSLIDE TYPES =====")
print(gdf["Name"].value_counts())


# ============================================================
# 5. STUDY AREA / EXTENT
# ============================================================

print("\n===== STUDY AREA =====")
print(gdf["Extent"].value_counts(dropna=False))


# ============================================================
# 6. COORDINATE SYSTEM
# ============================================================

print("\n===== COORDINATE SYSTEM =====")
print(gdf.crs)


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n===== MISSING VALUES =====")
print(gdf.isnull().sum())