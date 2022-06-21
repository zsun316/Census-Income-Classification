import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


def load_model(root_path, model_name):
    with open(os.path.join(root_path, "model", model_name), "rb") as f:
        model = joblib.load(f)
    return model


def save_model(model, root_path, model_name):
    with open(os.path.join(root_path, "model", model_name), "wb") as f:
        joblib.dump(model, f)


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    lgb_model = lgb.LGBMClassifier()
    param_grid = {
        'num_leaves': [8, 16, 31, 50, 100, 1000],
        'max_depth': [3, 5, 7, 10],
        'reg_alpha': [0.001, 0.01, 0.05, 0.1]
    }

    lgb_cv = GridSearchCV(estimator=lgb_model,
                          param_grid=param_grid,
                          n_jobs=-1)
    lgb_cv.fit(X_train, y_train)

    return lgb_cv.best_estimator_, lgb_cv.best_params_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred


def compute_slice_metrics(X_test,
                          y_test,
                          y_pred,
                          cat_feats):
    """

    Parameters
    ----------
    X_test: test data to be used for cal metrics
    y_test: true labels.
    y_pred: predictions
    cat_feats: categorical features

    Returns
    -------

    """
    y_test = pd.Series(np.squeeze(y_test))
    y_pred = pd.Series(np.squeeze(y_pred))

    df_test = pd.concat([X_test, y_test, y_pred], axis=1)
    df_test.columns = list(X_test.columns) + ['y_test', 'y_pred']

    TP = df_test[df_test['y_test'] == 1].groupby(cat_feats).sum()
    FN = df_test[df_test['y_test'] == 1].groupby(cat_feats).apply(lambda x: x.count() - x.sum())
    FP = df_test[df_test['y_test'] == 0].groupby(cat_feats).sum()

    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f_score = 2/(1/precision + 1/recall)

    slice_metrics = pd.concat([precision, recall, f_score], axis=1)
    slice_metrics.columns = ['precision', 'recall', 'f_score']

    return slice_metrics




