import geopandas as gpd
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD THE CLEANED LANDSLIDE DATA
# ============================================================

file_path = "data/processed/sikkim_landslides_cleaned.geojson"

gdf = gpd.read_file(file_path)

print("Number of landslide records:", len(gdf))
print("Coordinate system:", gdf.crs)


# ============================================================
# 2. CHECK THE GEOGRAPHIC EXTENT
# ============================================================

print("\n===== GEOGRAPHIC EXTENT =====")

print("Minimum longitude:", gdf.geometry.x.min())
print("Maximum longitude:", gdf.geometry.x.max())
print("Minimum latitude:", gdf.geometry.y.min())
print("Maximum latitude:", gdf.geometry.y.max())


# ============================================================
# 3. PLOT LANDSLIDE LOCATIONS
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))

gdf.plot(
    ax=ax,
    markersize=8,
    alpha=0.7
)

ax.set_title("Southern Sikkim Landslide Inventory")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()