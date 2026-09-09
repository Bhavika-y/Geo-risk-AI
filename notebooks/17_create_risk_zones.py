import rasterio
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD SUSCEPTIBILITY MAP
# ============================================================

input_file = "outputs/sikkim_landslide_susceptibility.tif"

with rasterio.open(input_file) as src:

    risk = src.read(1)

    profile = src.profile.copy()

    print("===== SUSCEPTIBILITY MAP =====")
    print("Rows:", src.height)
    print("Columns:", src.width)
    print("CRS:", src.crs)


# ============================================================
# 2. CREATE RISK ZONES
# ============================================================

risk_zones = np.full(
    risk.shape,
    np.nan,
    dtype=np.float32
)

valid = np.isfinite(risk)


# Low Risk = 1
risk_zones[
    valid & (risk < 0.25)
] = 1


# Moderate Risk = 2
risk_zones[
    valid & (risk >= 0.25) & (risk < 0.50)
] = 2


# High Risk = 3
risk_zones[
    valid & (risk >= 0.50) & (risk < 0.75)
] = 3


# Very High Risk = 4
risk_zones[
    valid & (risk >= 0.75)
] = 4


# ============================================================
# 3. PRINT AREA DISTRIBUTION
# ============================================================

print("\n===== RISK ZONE DISTRIBUTION =====")

for zone in [1, 2, 3, 4]:

    count = np.sum(risk_zones == zone)

    percentage = (
        count / np.sum(valid)
    ) * 100

    print(
        f"Zone {zone}: {count:,} pixels "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 4. SAVE RISK ZONE GEOTIFF
# ============================================================

output_file = "outputs/sikkim_risk_zones.tif"

profile.update(
    dtype="float32",
    count=1,
    nodata=np.nan
)

with rasterio.open(
    output_file,
    "w",
    **profile
) as dst:

    dst.write(
        risk_zones,
        1
    )


# ============================================================
# 5. CREATE VISUAL MAP
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    risk_zones,
    vmin=1,
    vmax=4
)

plt.title(
    "Southern Sikkim Landslide Risk Zones"
)

plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

cbar = plt.colorbar(
    ticks=[1, 2, 3, 4]
)

cbar.ax.set_yticklabels([
    "Low",
    "Moderate",
    "High",
    "Very High"
])

plt.tight_layout()

plt.savefig(
    "outputs/sikkim_risk_zones.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 6. FINISHED
# ============================================================

print("\n===== COMPLETE =====")

print(
    "GeoTIFF saved:"
)

print(
    "outputs/sikkim_risk_zones.tif"
)

print(
    "Image saved:"
)

print(
    "outputs/sikkim_risk_zones.png"
)