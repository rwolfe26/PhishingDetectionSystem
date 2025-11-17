import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
model = tf.keras.models.load_model("phish_tf_model.keras")
print("Model loaded ✔")
model.summary()

# Load the dataset
df = pd.read_csv("Phishing_Legitimate_full.csv")

# Split features/labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Must use the SAME scaler used during training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predict on first 5 samples
preds = model.predict(X_scaled[:5])
print("Predictions:", preds)