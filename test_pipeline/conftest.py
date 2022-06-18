import pytest
import os
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


@pytest.fixture(scope='session')
def data():
    """
        Get dataset
    """
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    file_name = "census.csv"
    file_path = os.path.join(root_path, "data", file_name)

    df = pd.read_csv(file_path, low_memory=False)
    return df

