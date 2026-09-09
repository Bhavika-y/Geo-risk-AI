import rasterio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.warp import calculate_default_transform, reproject, Resampling


# --------------------------------------------------
# 1. DEM FILE
# --------------------------------------------------

dem_file = r"C:\Users\HP\Downloads\sikkim_dem.tif.tif"


# --------------------------------------------------
# 2. OPEN DEM AND REPROJECT TO UTM
# --------------------------------------------------

with rasterio.open(dem_file) as src:

    print("===== ORIGINAL DEM =====")
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)

    # Target CRS: UTM Zone 45N
    target_crs = "EPSG:32645"

    transform, width, height = calculate_default_transform(
        src.crs,
        target_crs,
        src.width,
        src.height,
        *src.bounds
    )

    elevation = np.empty((height, width), dtype=np.float32)

    reproject(
        source=rasterio.band(src, 1),
        destination=elevation,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )

    resolution_x = transform.a
    resolution_y = abs(transform.e)


print("\n===== REPROJECTED DEM =====")
print("CRS:", target_crs)
print("Pixel resolution:", resolution_x, "x", resolution_y, "meters")


# --------------------------------------------------
# 3. HANDLE INVALID ELEVATION VALUES
# --------------------------------------------------

elevation[elevation < -1000] = np.nan


# --------------------------------------------------
# 4. CALCULATE TERRAIN GRADIENT
# --------------------------------------------------

dy, dx = np.gradient(
    elevation,
    resolution_y,
    resolution_x
)


# --------------------------------------------------
# 5. CALCULATE SLOPE
# --------------------------------------------------

slope = np.degrees(
    np.arctan(
        np.sqrt(dx**2 + dy**2)
    )
)


# --------------------------------------------------
# 6. CALCULATE ASPECT
# --------------------------------------------------

aspect = np.degrees(
    np.arctan2(dx, -dy)
)

aspect = (aspect + 360) % 360


# --------------------------------------------------
# 7. CALCULATE CURVATURE
# --------------------------------------------------

d2x = np.gradient(dx, resolution_x, axis=1)
d2y = np.gradient(dy, resolution_y, axis=0)

curvature = d2x + d2y


# --------------------------------------------------
# 8. PRINT TERRAIN STATISTICS
# --------------------------------------------------

print("\n===== TERRAIN FEATURES =====")

print("\nElevation:")
print("Minimum:", np.nanmin(elevation), "m")
print("Maximum:", np.nanmax(elevation), "m")
print("Mean:", np.nanmean(elevation), "m")

print("\nSlope:")
print("Minimum:", np.nanmin(slope), "degrees")
print("Maximum:", np.nanmax(slope), "degrees")
print("Mean:", np.nanmean(slope), "degrees")

print("\nAspect:")
print("Minimum:", np.nanmin(aspect), "degrees")
print("Maximum:", np.nanmax(aspect), "degrees")

print("\nCurvature:")
print("Minimum:", np.nanmin(curvature))
print("Maximum:", np.nanmax(curvature))


# --------------------------------------------------
# 9. PLOT SLOPE
# --------------------------------------------------

plt.figure(figsize=(10, 8))

plt.imshow(slope)

plt.title("Slope Derived from DEM")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")
plt.colorbar(label="Slope (degrees)")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. PLOT ASPECT
# --------------------------------------------------

plt.figure(figsize=(10, 8))

plt.imshow(aspect)

plt.title("Aspect Derived from DEM")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")
plt.colorbar(label="Aspect (degrees)")

plt.tight_layout()
plt.show()