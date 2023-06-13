import argparse
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

from utils import load_videos, parse_arguments, get_path_config_from_args, print_performance_metrics
import pandas as pd

import xgboost as xgb

def main():
    config = get_path_config_from_args()
    videos = load_videos(config)
    videos_with_features = [v for v in videos if v.has_features()]
    print(f'Loaded {len(videos_with_features)} videos with features out of a total of {len(videos)} videos.')

    metadata = pd.read_csv(config.meta_csv_path)
    ids = [v.id for v in videos_with_features]


    X = np.stack([v.features for v in videos_with_features])

    # metadata.set_index('VideoID')
    filtered_metadata = metadata[metadata['VideoID'].isin(ids)]
    sorted_metadata = filtered_metadata.sort_values(by=['VideoID'],
                                                    key=lambda x: pd.Categorical(x, categories=ids, ordered=True))
    y = sorted_metadata['Disgust category']

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.33, random_state=0)
    X_validation, X_test, y_validation, y_test = train_test_split(X_rest, y_rest, test_size=0.50, random_state=0)
    print(f'Split {len(y)} samples into the following sets:')
    print(f'Train set {len(y_train)}')
    print(f'Validation set {len(y_validation)}')
    print(f'Test set {len(y_test)}')

    random.seed(0)
    # predicted = train_and_predict_using_simple_default_rf(X_train, X_validation, y_train)
    # predicted = train_and_predict_using_grid_search_rf(X_train, X_validation, y_train)
    predicted = train_and_predict_using_xgboost(X_train, X_validation, y_train)
    print_performance_metrics(trues=y_validation, predicted=predicted, class_list=y_train.unique())
    print(confusion_matrix(y_validation, predicted), 'true pathogen disgust:', len([p for p in predicted if p == 'pathogen disgust']))


def train_and_predict_using_simple_default_rf(X_train, X_validation, y_train):
    n_trees = 200
    clf = RandomForestClassifier(n_estimators=n_trees)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_validation)
    return predicted

def train_and_predict_using_xgboost(X_train, X_validation, y_train):
    n_trees = 200
    clf = xgb.XGBClassifier(n_estimators=n_trees)

    y_train_xgboost = [0 if v == 'moral disgust' else 1 for v in y_train]
    clf.fit(X_train, y_train_xgboost)

    predicted = clf.predict(X_validation)
    return ['moral disgust' if v == 0 else 'pathogen disgust' for v in predicted]

def train_and_predict_using_grid_search_rf(X_train, X_validation, y_train):
    # Define the hyperparameters you want to tune
    param_grid = {
        # 'n_estimators': [100, 200, 300, 500],  # number of trees
        'max_depth': [10, 20, 30, None],  # max number of levels in each decision tree
        # 'min_samples_split': [2, 5, 10],  # min number of samples required to split a node
        # 'min_samples_leaf': [1, 2, 4],  # min number of samples required at each leaf node
        'bootstrap': [True, False]  # method for sampling data points (with or without replacement)
    }

    # Initialize the base classifier
    clf = RandomForestClassifier()

    # Initialize the grid search object

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)

    # Fit the grid search object to the data
    grid_search.fit(X_train, y_train)

    # Get the optimal hyperparameters
    best_params = grid_search.best_params_

    print(f"Best parameters: {best_params}")

    # You can now create a new random forest classifier with the best parameters
    clf_best = RandomForestClassifier(**best_params)
    clf_best.fit(X_train, y_train)

    # And then use this classifier for your prediction
    predicted = clf_best.predict(X_validation)
    return predicted


if __name__ == '__main__':
    main()
