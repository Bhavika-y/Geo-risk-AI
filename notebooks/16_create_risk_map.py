import rasterio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DEM
# ============================================================

dem_file = r"C:\Users\HP\Downloads\sikkim_dem.tif.tif"

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

    profile = src.profile.copy()


# ============================================================
# 2. CALCULATE TERRAIN FEATURES
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


# ============================================================
# 3. LOAD RAINFALL
# ============================================================

rainfall_file = r"C:\Users\HP\Downloads\chirps-v2.0.2019.tif"

with rasterio.open(rainfall_file) as rain:

    rainfall = np.empty(
        (height, width),
        dtype=np.float32
    )

    reproject(
        source=rasterio.band(rain, 1),
        destination=rainfall,
        src_transform=rain.transform,
        src_crs=rain.crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )


# ============================================================
# 4. CREATE VALID PIXEL MASK
# ============================================================

valid = (
    np.isfinite(elevation) &
    np.isfinite(slope) &
    np.isfinite(aspect) &
    np.isfinite(curvature) &
    np.isfinite(rainfall)
)

print("Valid pixels:", valid.sum())


# ============================================================
# 5. PREPARE MODEL TRAINING DATA
# ============================================================

X = np.column_stack([
    elevation[valid],
    slope[valid],
    aspect[valid],
    curvature[valid],
    rainfall[valid]
])

print("Prediction pixels:", len(X))


# ============================================================
# 6. LOAD TRAINING DATA
# ============================================================

import pandas as pd

df = pd.read_csv(
    "data/processed/sikkim_final_ml_dataset.csv"
)

features = [
    "Elevation",
    "Slope",
    "Aspect",
    "Curvature",
    "rainfall_2019"
]

X_train = df[features]
y_train = df["landslide"]


# ============================================================
# 7. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print("Random Forest trained.")


# ============================================================
# 8. PREDICT LANDSLIDE PROBABILITY
# ============================================================

print("Generating susceptibility map...")

X_prediction = pd.DataFrame(X, columns=features)

probabilities = model.predict_proba(X_prediction)[:, 1]

risk_map = np.full(
    elevation.shape,
    np.nan,
    dtype=np.float32
)

risk_map[valid] = probabilities


# ============================================================
# 9. DISPLAY MAP
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    risk_map,
    vmin=0,
    vmax=1
)

plt.title(
    "Southern Sikkim Landslide Susceptibility Map"
)

plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

plt.colorbar(
    label="Landslide Susceptibility"
)

plt.tight_layout()

plt.savefig(
    "outputs/sikkim_landslide_susceptibility.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. SAVE GEOTIFF
# ============================================================

profile.update(
    driver="GTiff",
    height=risk_map.shape[0],
    width=risk_map.shape[1],
    count=1,
    dtype="float32",
    crs=target_crs,
    transform=transform,
    nodata=np.nan
)

with rasterio.open(
    "outputs/sikkim_landslide_susceptibility.tif",
    "w",
    **profile
) as dst:

    dst.write(
        risk_map,
        1
    )

print("\n===== COMPLETE =====")

print(
    "Map saved to:"
)

print(
    "outputs/sikkim_landslide_susceptibility.png"
)

print(
    "outputs/sikkim_landslide_susceptibility.tif"
)