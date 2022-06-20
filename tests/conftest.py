import pytest
import os
import numpy as np
import pandas as pd
import joblib

from fastapi.testclient import TestClient
# from ..ml_pipeline.ml.data import load_data


@pytest.fixture(scope='session')
def data():
    """
        Get dataset
    """
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    file_name = "preprocessed_census.csv"
    file_path = os.path.join(root_path, "data", file_name)

    df = pd.read_csv(file_path, low_memory=False)
    return df


def model():
    """
        Get Model
    Returns
    -------

    """
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    model_name = "model.pkl"

    trained_model = joblib.load(os.path.join(root_path, "model", model_name))

    return trained_model
