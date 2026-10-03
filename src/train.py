import os
import numpy as np
import yaml
import tensorflow as tf

with open("params.yaml") as f:
    params = yaml.safe_load(f)

p = params["train"]

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")

x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(p["dense_units"], activation="relu"),
    tf.keras.layers.Dropout(p["dropout_rate"]),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=p["learning_rate"]
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=p["epochs"],
    batch_size=p["batch_size"]
)

os.makedirs("models", exist_ok=True)

model.save("models/model.h5")

import pandas as pd
pd.DataFrame(history.history).to_csv(
    "models/history.csv",
    index=False
)

print("Model trained.")