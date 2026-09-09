import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# 1. LOAD DATA
# ==============================

file_path = "data/processed/sikkim_final_ml_dataset.csv"

df = pd.read_csv(file_path)

print("Total samples:", len(df))


# ==============================
# 2. SELECT FEATURES
# ==============================

features = [
    "Elevation",
    "Slope",
    "Aspect",
    "Curvature",
    "rainfall_2019"
]

X = df[features]

y = df["landslide"]


# ==============================
# 3. SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 4. SCALE FEATURES
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==============================
# 5. LOGISTIC REGRESSION
# ==============================

logistic_model = LogisticRegression(
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test_scaled
)


# ==============================
# 6. RANDOM FOREST
# ==============================

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

rf_predictions = random_forest.predict(
    X_test
)


# ==============================
# 7. MODEL EVALUATION
# ==============================

print("\n===== LOGISTIC REGRESSION =====")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        logistic_predictions
    )
)

print(
    classification_report(
        y_test,
        logistic_predictions
    )
)


print("\n===== RANDOM FOREST =====")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        rf_predictions
    )
)

print(
    classification_report(
        y_test,
        rf_predictions
    )
)