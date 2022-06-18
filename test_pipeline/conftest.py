import pytest
import os
import numpy as np
import pandas as pd
from ml_pipeline.ml.data import load_data
from fastapi.testclient import TestClient


@pytest.fixture(scope='session')
def data():
    """
    Get dataset
    """
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    file_name = "census.csv"
    file_path = os.path.join(root_path, "data")

    return load_data(file_path, 'census.csv')

