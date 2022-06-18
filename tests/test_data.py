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

    for col, format_verification_func in required_quant_columns.items():
        assert format_verification_func(data[col]), f"Column {col} failed test {format_verification_func}"





