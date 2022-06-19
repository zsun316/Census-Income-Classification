import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


def load_data(root_path, file_name):
    """

    Parameters
    ----------
    root_path: path of the root dir
    file_name: file name

    Returns: either raw df or preprocessed df
    -------

    """
    path = os.path.join(root_path, "data", file_name)
    df = pd.read_csv(path, low_memory=False)

    return df


def save_data(df: pd.DataFrame, root_path, file_name):
    """

    Parameters
    ----------
    df:          pd.DataFrame to be saved.
    root_path:   path of the root dir
    file_name:   file name

    Returns:     None
    -------

    """
    df.to_csv(os.path.join(root_path, 'data', file_name))


def process_data(
        X,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        lb=None,
        preprocessor=None,
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    # X_categorical = X[categorical_features].values
    # X_continuous = X.drop(*[categorical_features], axis=1)

    continuous_features = list(set(X.columns) - set(categorical_features))
    if training is True:
        preprocessor = ColumnTransformer(
            transformers=[
                ("continuous_feats", StandardScaler(), continuous_features),
                ("non_ordinal_cat", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_features)
            ],
            remainder="drop"  # This drops the columns that we do not transform
        )

        lb = LabelBinarizer()
        X = preprocessor.fit_transform(X)
        y = lb.fit_transform(y.values).ravel()
    else:
        X = preprocessor.transform(X)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    # X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, preprocessor, lb


if __name__ == "__main__":
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    file_name = "preprocessed_census.csv"
    proc_df = load_data(root_path, file_name)

    cat_feats = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country'
    ]

    X, y, preprocessor, lb = process_data(proc_df, cat_feats, 'salary')