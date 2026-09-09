import geopandas as gpd


# ==============================
# 1. LOAD DATA
# ==============================

input_file = "data/processed/sikkim_samples_with_rainfall.geojson"

gdf = gpd.read_file(input_file)

print("Total samples:", len(gdf))


# ==============================
# 2. SELECT ML FEATURES
# ==============================

features = [
    "Elevation",
    "Slope",
    "Aspect",
    "Curvature",
    "rainfall_2019"
]

target = "landslide"


# ==============================
# 3. CREATE ML DATAFRAME
# ==============================

ml_data = gdf[features + [target]].copy()


# ==============================
# 4. CHECK MISSING VALUES
# ==============================

print("\n===== MISSING VALUES =====")

print(ml_data.isna().sum())


# ==============================
# 5. REMOVE INCOMPLETE ROWS
# ==============================

ml_data = ml_data.dropna()

print("\nSamples after removing missing values:",
      len(ml_data))


# ==============================
# 6. CHECK CLASS BALANCE
# ==============================

print("\n===== CLASS DISTRIBUTION =====")

print(ml_data[target].value_counts())


# ==============================
# 7. FEATURE STATISTICS
# ==============================

print("\n===== FEATURE STATISTICS =====")

print(ml_data[features].describe())


# ==============================
# 8. SAVE FINAL ML DATASET
# ==============================

output_file = "data/processed/sikkim_final_ml_dataset.csv"

ml_data.to_csv(
    output_file,
    index=False
)

print("\nFinal ML dataset saved to:")
print(output_file)