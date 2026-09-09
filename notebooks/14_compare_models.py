import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling
)


# ============================================================
# 1. FILE PATHS
# ============================================================

points_file = (
    "data/processed/"
    "sikkim_landslide_binary_samples_v2.geojson"
)

dem_file = r"C:\Users\HP\Downloads\sikkim_dem.tif.tif"

rainfall_file = r"C:\Users\HP\Downloads\chirps-v2.0.2019.tif"

output_file = (
    "data/processed/"
    "sikkim_final_ml_dataset.csv"
)


# ============================================================
# 2. LOAD 370 LANDSLIDE / NON-LANDSLIDE POINTS
# ============================================================

gdf = gpd.read_file(points_file)

print("===== ORIGINAL DATA =====")
print("Total samples:", len(gdf))
print("\nClass distribution:")
print(gdf["landslide"].value_counts())


# ============================================================
# 3. OPEN DEM
# ============================================================

with rasterio.open(dem_file) as src:

    print("\n===== DEM =====")
    print("CRS:", src.crs)
    print("Resolution:", src.res)

    target_crs = "EPSG:32645"

    transform, width, height = calculate_default_transform(
        src.crs,
        target_crs,
        src.width,
        src.height,
        *src.bounds
    )

    elevation = np.empty(
        (height, width),
        dtype=np.float32
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=elevation,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )


# ============================================================
# 4. DERIVE TERRAIN FEATURES
# ============================================================

resolution_x = transform.a
resolution_y = abs(transform.e)

elevation[elevation < -1000] = np.nan


dy, dx = np.gradient(
    elevation,
    resolution_y,
    resolution_x
)


slope = np.degrees(
    np.arctan(
        np.sqrt(dx**2 + dy**2)
    )
)


aspect = np.degrees(
    np.arctan2(dx, -dy)
)

aspect = (aspect + 360) % 360


d2x = np.gradient(
    dx,
    resolution_x,
    axis=1
)

d2y = np.gradient(
    dy,
    resolution_y,
    axis=0
)

curvature = d2x + d2y


print("\n===== TERRAIN FEATURES CREATED =====")

print(
    "Elevation:",
    np.nanmin(elevation),
    "to",
    np.nanmax(elevation)
)

print(
    "Slope:",
    np.nanmin(slope),
    "to",
    np.nanmax(slope)
)


# ============================================================
# 5. CONVERT POINTS TO DEM CRS
# ============================================================

gdf_utm = gdf.to_crs(target_crs)


# ============================================================
# 6. FUNCTION TO EXTRACT RASTER VALUES
# ============================================================

coordinates = [
    (point.x, point.y)
    for point in gdf_utm.geometry
]


def extract_values(raster):

    values = []

    for x, y in coordinates:

        col, row = ~transform * (x, y)

        col = int(col)
        row = int(row)

        if (
            0 <= row < raster.shape[0]
            and 0 <= col < raster.shape[1]
        ):

            values.append(
                raster[row, col]
            )

        else:

            values.append(np.nan)

    return values


# ============================================================
# 7. ADD TERRAIN FEATURES TO ALL 370 POINTS
# ============================================================

gdf_utm["Elevation"] = extract_values(
    elevation
)

gdf_utm["Slope"] = extract_values(
    slope
)

gdf_utm["Aspect"] = extract_values(
    aspect
)

gdf_utm["Curvature"] = extract_values(
    curvature
)


print("\n===== TERRAIN EXTRACTION COMPLETE =====")


# ============================================================
# 8. EXTRACT RAINFALL
# ============================================================

with rasterio.open(rainfall_file) as rain:

    print("\n===== RAINFALL =====")
    print("CRS:", rain.crs)
    print("Resolution:", rain.res)

    gdf_rain = gdf_utm.to_crs(
        rain.crs
    )

    rain_coordinates = [
        (point.x, point.y)
        for point in gdf_rain.geometry
    ]

    rainfall_values = []

    for value in rain.sample(
        rain_coordinates
    ):

        rainfall_values.append(
            value[0]
        )


gdf_utm["rainfall_2019"] = (
    rainfall_values
)


# ============================================================
# 9. CREATE FINAL ML TABLE
# ============================================================

features = [
    "Elevation",
    "Slope",
    "Aspect",
    "Curvature",
    "rainfall_2019"
]

target = "landslide"


ml_data = gdf_utm[
    features + [target]
].copy()


# ============================================================
# 10. CHECK MISSING VALUES
# ============================================================

print("\n===== MISSING VALUES =====")

print(
    ml_data.isna().sum()
)


# ============================================================
# 11. REMOVE INCOMPLETE ROWS
# ============================================================

ml_data = ml_data.dropna()

print(
    "\nSamples after removing missing values:",
    len(ml_data)
)


# ============================================================
# 12. CHECK CLASS BALANCE
# ============================================================

print("\n===== FINAL CLASS DISTRIBUTION =====")

print(
    ml_data["landslide"].value_counts()
)


# ============================================================
# 13. FEATURE STATISTICS
# ============================================================

print("\n===== FEATURE STATISTICS =====")

print(
    ml_data[features].describe()
)


# ============================================================
# 14. SAVE FINAL ML DATASET
# ============================================================

ml_data.to_csv(
    output_file,
    index=False
)

print("\n===== SUCCESS =====")

print(
    "Final dataset saved to:"
)

print(output_file)