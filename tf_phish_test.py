import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split #preparing data for model training
from sklearn.preprocessing import StandardScaler #preprocessing
import numpy as np #Linear algebra component 

# 1. Load the dataset
df = pd.read_csv("Phishing_Legitimate_full.csv")

# Last column is usually the target (phishing/legitimate)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. Train/test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# 3. Normalize - important for NN
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. Build super baseline model
model = tf.keras.Sequential([
    tf.keras.Input(shape = (x.shape[1],)), # Fixed error with the shape of data
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train
history = model.fit(
    x_train,y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# 6. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Test accuracy:", acc)

# 7. Save model to test loading/inference
model.save("phish_tf_model.keras")
print("Model saved")