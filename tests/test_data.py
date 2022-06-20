from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pytest


def test_column_presence_and_type(data):
    required_cat_columns = {
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country'
    }

    required_quant_columns = {
        'age': pd.api.types.is_integer_dtype,
        'fnlgt': pd.api.types.is_integer_dtype,
        'capital-gain': pd.api.types.is_integer_dtype,
        'hours-per-week': pd.api.types.is_integer_dtype,
        'capital-loss': pd.api.types.is_integer_dtype,
    }

    # Check cat column presence
    assert set(data.columns.values).issuperset(required_cat_columns), "Missing required categorical variable"

    # Check continuous column presence
    assert set(data.columns.values).issuperset(set(required_quant_columns.keys())), "Missing required quant variable"

    # Check continuous column data types
    for col, format_verification_func in required_quant_columns.items():
        assert format_verification_func(data[col]), f"Column {col} failed test {format_verification_func}"
    # print("--------------", 'test_column_presence_and_type passed', "--------------")


def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."
    # print("--------------", 'test_data_shape passed', "--------------")


def test_inference_process(metrics):
    assert metrics['precision'] > 0.5, "Invalid precision"
    assert metrics['recall'] > 0.5, "Invalid recall"
    assert metrics['fbeta'] > 0.5, "Invalid f-beta score"


