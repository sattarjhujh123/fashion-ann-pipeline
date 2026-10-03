import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_size = params["preprocess"]["test_size"]
seed = params["preprocess"]["seed"]

os.makedirs("data/processed", exist_ok=True)

x_train = np.load("data/raw/x_train.npy") / 255.0
y_train = np.load("data/raw/y_train.npy")

x_test = np.load("data/raw/x_test.npy") / 255.0
y_test = np.load("data/raw/y_test.npy")

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=test_size,
    random_state=seed
)

np.save("data/processed/x_train.npy", x_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/x_val.npy", x_val)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/x_test.npy", x_test)
np.save("data/processed/y_test.npy", y_test)

print("Processed data saved.")