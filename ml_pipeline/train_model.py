# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import load_data, save_data, process_data
from ml.model import save_model, train_model

import os
import yaml

# Add the necessary imports for the starter code.
root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# Add code to load in the data.
data = load_data(root_path, "clean_census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.

# Train and save a model.
model = train_model(X_train, y_train)
