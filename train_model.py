import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/Housing_Hamilton_Compressed.csv.gz")

print(df.head())
print(df.columns.tolist())

# -----------------------------
# Select features and target
# -----------------------------
features = ["CALC_ACRES", "LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE"]
target = "APPRAISED_VALUE"

df = df[features + [target]].copy()

# -----------------------------
# Remove missing values
# -----------------------------
df = df.dropna()

# -----------------------------
# Separate X and y
# -----------------------------
X = df[features]
y = df[target]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# -----------------------------
# Scale Inputs
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Build Neural Network
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# -----------------------------
# Evaluate Model
# -----------------------------
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)

print("Test MAE:", mae)

# -----------------------------
# Save Artifacts
# -----------------------------
os.makedirs("artifacts", exist_ok=True)

model.save("artifacts/housing_model.h5")
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(features, "artifacts/feature_names.pkl")

print("Model and artifacts saved successfully.")
