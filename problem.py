import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Chicken meat authenticity classification'
_prediction_label_names = ["CONV", "1 Star", "ORG", "2 stars", "STD", "FR", "CF", "MAR"]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.Accuracy(name='acc', precision=3),
    rw.score_types.NegativeLogLikelihood(name='nll', precision=3),
]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=57)
    return cv.split(X, y)

_target_column_name = 'Production_system'
_ignore_column_names = ['Sample_number', 'Scan_type', 'Freshness']

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].to_numpy()
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    X_df = X_df.to_numpy()
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)