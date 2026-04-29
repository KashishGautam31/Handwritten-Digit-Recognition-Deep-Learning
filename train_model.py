# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
train = pd.read_csv("data/mnist_train.csv")
test = pd.read_csv("data/mnist_test.csv")

# Split features and labels
x_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

x_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape into 28x28 images
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Save model
model.save("model/digit_model.h5")

print("✅ Model trained and saved!")