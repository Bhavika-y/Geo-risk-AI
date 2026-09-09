import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "data/processed/sikkim_final_ml_dataset.csv"

df = pd.read_csv(file_path)

print("===== DATASET =====")
print("Total samples:", len(df))
print("\nClass distribution:")
print(df["landslide"].value_counts())


# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "Elevation",
    "Slope",
    "Aspect",
    "Curvature",
    "rainfall_2019"
]

X = df[features]
y = df["landslide"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nRandom Forest trained successfully.")


# ============================================================
# 5. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


# ============================================================
# 6. HANDLE SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):
    shap_values_plot = shap_values[1]
else:
    shap_values_plot = shap_values


# ============================================================
# 7. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\n===== SHAP FEATURE IMPORTANCE =====")

importance = pd.DataFrame({
    "Feature": features,
    "Mean_Absolute_SHAP": abs(shap_values_plot).mean(axis=0)
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print(importance)


# ============================================================
# 8. SHAP SUMMARY PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values_plot,
    X_test,
    show=False
)

plt.title("SHAP Feature Importance - Landslide Risk")

plt.tight_layout()

plt.savefig(
    "outputs/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. BAR IMPORTANCE PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.barh(
    importance["Feature"],
    importance["Mean_Absolute_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Feature Importance for Landslide Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "outputs/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. SAVE IMPORTANCE TABLE
# ============================================================

importance.to_csv(
    "outputs/shap_feature_importance.csv",
    index=False
)

print("\nSHAP analysis completed.")

print("\nFiles saved:")
print("outputs/shap_summary.png")
print("outputs/shap_feature_importance.png")
print("outputs/shap_feature_importance.csv")