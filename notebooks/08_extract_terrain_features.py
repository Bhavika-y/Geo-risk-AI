import geopandas as gpd
import rasterio
import numpy as np

from rasterio.warp import calculate_default_transform, reproject, Resampling


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

landslide_file = "data/processed/sikkim_landslides_cleaned.geojson"
dem_file =r"C:\Users\HP\Downloads\sikkim_dem.tif.tif"

output_file = "data/processed/sikkim_landslides_with_terrain.geojson"


# --------------------------------------------------
# 2. LOAD LANDSLIDE DATA
# --------------------------------------------------

gdf = gpd.read_file(landslide_file)

print("Number of landslide points:", len(gdf))
print("Original CRS:", gdf.crs)


# --------------------------------------------------
# 3. OPEN DEM
# --------------------------------------------------

with rasterio.open(dem_file) as src:

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


# --------------------------------------------------
# 4. CALCULATE TERRAIN FEATURES
# --------------------------------------------------

resolution_x = transform.a
resolution_y = abs(transform.e)

elevation[elevation < -1000] = np.nan

dy, dx = np.gradient(
    elevation,
    resolution_y,
    resolution_x
)


# SLOPE

slope = np.degrees(
    np.arctan(
        np.sqrt(dx**2 + dy**2)
    )
)


# ASPECT

aspect = np.degrees(
    np.arctan2(dx, -dy)
)

aspect = (aspect + 360) % 360


# CURVATURE

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


# --------------------------------------------------
# 5. CONVERT LANDSLIDE POINTS TO UTM
# --------------------------------------------------

gdf_utm = gdf.to_crs(target_crs)

print("Converted landslide points to:", target_crs)


# --------------------------------------------------
# 6. EXTRACT RASTER VALUES
# --------------------------------------------------

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
            values.append(raster[row, col])

        else:
            values.append(np.nan)

    return values


gdf_utm["dem_elevation"] = extract_values(elevation)
gdf_utm["dem_slope"] = extract_values(slope)
gdf_utm["dem_aspect"] = extract_values(aspect)
gdf_utm["dem_curvature"] = extract_values(curvature)


# --------------------------------------------------
# 7. CHECK RESULTS
# --------------------------------------------------

print("\n===== EXTRACTED TERRAIN FEATURES =====")

print(
    gdf_utm[
        [
            "dem_elevation",
            "dem_slope",
            "dem_aspect",
            "dem_curvature"
        ]
    ].describe()
)


print("\nMissing terrain values:")

print(
    gdf_utm[
        [
            "dem_elevation",
            "dem_slope",
            "dem_aspect",
            "dem_curvature"
        ]
    ].isna().sum()
)


# --------------------------------------------------
# 8. CONVERT BACK TO WGS84
# --------------------------------------------------

gdf_final = gdf_utm.to_crs("EPSG:4326")


# --------------------------------------------------
# 9. SAVE DATASET
# --------------------------------------------------

gdf_final.to_file(
    output_file,
    driver="GeoJSON"
)

print("\nDataset saved to:")
print(output_file)