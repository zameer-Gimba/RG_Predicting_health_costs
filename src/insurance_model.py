# Healthcare Insurance Cost Prediction using Neural Networks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset (local path)
df = pd.read_csv("data/insurance.csv")

# Feature groups
categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]
)

# Features and target
X = df.drop(columns=["expenses"])
y = df["expenses"]

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Neural network model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

# Compile
model.compile(
    optimizer="adam",
    loss="mae",
    metrics=["mae"]
)

# Train (do NOT change epochs or validation split)
epochs = 100
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    validation_split=0.2,
    verbose=0
)

# Evaluate
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Mean Absolute Error: {mae}")

# Predictions
y_pred = model.predict(X_test)

# Visualization (same default matplotlib behavior)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")
plt.title("Actual vs Predicted Healthcare Costs")
plt.show()
