import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# 1. LOAD FINAL ML DATASET
# ==============================

file_path = "data/processed/sikkim_final_ml_dataset.csv"

df = pd.read_csv(file_path)

print("Total samples:", len(df))
print("Features:", df.columns.tolist())


# ==============================
# 2. CLASS DISTRIBUTION
# ==============================

print("\n===== CLASS DISTRIBUTION =====")

print(df["landslide"].value_counts())

plt.figure(figsize=(7, 5))

df["landslide"].value_counts().plot(
    kind="bar"
)

plt.title("Landslide vs Non-Landslide Samples")
plt.xlabel("Class (0 = Non-Landslide, 1 = Landslide)")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.show()


# ==============================
# 3. FEATURE DISTRIBUTIONS
# ==============================

features = [
    "Elevation",
    "Slope",
    "Aspect",
    "Curvature",
    "rainfall_2019"
]

for feature in features:

    plt.figure(figsize=(8, 5))

    df.boxplot(
        column=feature,
        by="landslide"
    )

    plt.title(f"{feature} vs Landslide")
    plt.suptitle("")

    plt.xlabel(
        "Class (0 = Non-Landslide, 1 = Landslide)"
    )

    plt.ylabel(feature)

    plt.tight_layout()
    plt.show()


# ==============================
# 4. CORRELATION MATRIX
# ==============================

correlation = df[
    features + ["landslide"]
].corr()

print("\n===== CORRELATION MATRIX =====")

print(correlation)


# ==============================
# 5. CORRELATION HEATMAP
# ==============================

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()
plt.show()


# ==============================
# 6. GROUP STATISTICS
# ==============================

print("\n===== AVERAGE VALUES BY CLASS =====")

print(
    df.groupby("landslide")[features].mean()
)