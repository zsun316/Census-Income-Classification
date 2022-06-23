import pytest
import os
import sys
import numpy as np
import pandas as pd
import joblib

from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from main import app


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


@pytest.fixture(scope='session')
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


@pytest.fixture(scope='session')
def metrics():
    """
        Get Model
    Returns
    -------

    """
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    metrics_name = "metrics.pkl"

    trained_metrics = joblib.load(os.path.join(root_path, "model", metrics_name))

    return trained_metrics


@pytest.fixture(scope='session')
def test_api_example():
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    file_name = "preprocessed_census.csv"
    file_path = os.path.join(root_path, "data", file_name)

    df = pd.read_csv(file_path, low_memory=False)

    zero_exp = df[df['salary'] == '<=50K'].iloc[0].drop('salary')
    one_exp = df[df['salary'] == '>50K'].iloc[0].drop('salary')

    return [zero_exp, one_exp]


@pytest.fixture(scope='session')
def client():
    """
    Get dataset
    """
    client = TestClient(app)
    return client
