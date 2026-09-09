import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LOCATE THE DEM FILE
# ============================================================

file_path =r"C:\Users\HP\Downloads\sikkim_dem.tif.tif"


# ============================================================
# 2. OPEN THE DEM
# ============================================================

with rasterio.open(file_path) as dem:

    print("===== DEM INFORMATION =====")

    print("File:", file_path)
    print("Coordinate system:", dem.crs)
    print("Number of columns:", dem.width)
    print("Number of rows:", dem.height)
    print("Number of bands:", dem.count)

    print("\nPixel resolution:")
    print(dem.res)

    print("\nGeographic extent:")
    print(dem.bounds)

    print("\nElevation range:")

    elevation = dem.read(1)

    print("Minimum elevation:", np.nanmin(elevation))
    print("Maximum elevation:", np.nanmax(elevation))


# ============================================================
# 3. CHECK FOR MISSING / INVALID PIXELS
# ============================================================

print("\n===== DATA QUALITY =====")

missing_pixels = np.isnan(elevation).sum()

print("Missing pixels:", missing_pixels)


# ============================================================
# 4. DISPLAY THE DEM
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(elevation)

plt.title("Southern Sikkim Digital Elevation Model")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

plt.colorbar(label="Elevation")

plt.tight_layout()
plt.show()