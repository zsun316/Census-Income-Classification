# Script to train machine learning model.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import load_data, save_data, process_data
from ml.model import save_model, train_model, compute_model_metrics, inference, compute_slice_metrics

import os

# Add the necessary imports for the starter code.
root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# Add code to load in the data.
data = load_data(root_path, "preprocessed_census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)


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

X_train, y_train, preprocessor, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    preprocessor=preprocessor,
    lb=lb
)

# Train a model.
model, params = train_model(X_train, y_train)
y_pred = inference(model, X_test)
precision, recall, f_beta = compute_model_metrics(y_test, y_pred)

slice_metrics = compute_slice_metrics(X_test, y_test, y_pred, cat_features)


print("best parameters", params)
print('precision:', precision)
print('recall', recall)
print('f_beta_score', f_beta)

# save data
save_data(train, root_path, 'train.csv')
save_data(test, root_path, 'test.csv')

save_data(slice_metrics, root_path, 'slice_metrics.csv')

# save model
save_model(model, root_path, "model.pkl")
save_model(preprocessor, root_path, "preprocessor.pkl")
save_model(lb, root_path, "lb.pkl")
save_model({'precision': precision,
            'recall': recall,
            'fbeta': f_beta}, root_path, 'metrics.pkl')
