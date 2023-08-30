from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from IPython.core.display_functions import display
from pycaret.classification import setup, compare_models, tune_model, automl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from tqdm import tqdm

from disgust.disgust_classes import class_name_to_class_id, class_id_to_class_name


def train_and_predict_using_rf(X_train, X_validation, y_train):
    best_params = {'bootstrap': False, 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10,
                   'n_estimators': 100}
    clf = RandomForestClassifier(**best_params)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_validation)
    probs = clf.predict_proba(X_validation)[:, 1]

    return predicted, probs, clf.feature_importances_


def train_and_predict_using_xgboost(X_train, X_validation, y_train):
    n_trees = 200
    clf = xgb.XGBClassifier(n_estimators=n_trees)

    y_train_xgboost = [class_name_to_class_id(class_name) for class_name in y_train]
    clf.fit(X_train, y_train_xgboost)

    predicted = clf.predict(X_validation)
    probs = clf.predict_proba(X_validation)[:, 1]

    return [class_id_to_class_name(class_id) for class_id in predicted], probs, clf.feature_importances_

def train_and_predict_using_pca_and_xgboost(X_train, X_validation, y_train):
    pca = PCA()
    pca = pca.fit(X_train)
    top_n_components = 250
    X_train_pc = pca.transform(X_train)[:, :top_n_components]
    X_validation_pc = pca.transform(X_validation)[:, :top_n_components]

    return train_and_predict_using_xgboost(X_train_pc, X_validation_pc, y_train)

def train_and_predict_using_pycaret(X_train, X_validation, y_train):
    data = pd.DataFrame(np.column_stack((X_train, y_train)), columns=list(range(X_train.shape[1]))+['disgust type'])
    pycaret_setup = setup(data, target='disgust type', session_id=123, log_experiment=True, experiment_name='disgust type')
    print(f'{pycaret_setup=}')
    best_model = compare_models()
    # tune_model(best_model)  # Couldn't improve
    # clf = automl()
    predicted = best_model.predict(X_validation)
    probs = best_model.predict_proba(X_validation)[:, 1]
    return [class_id_to_class_name(class_id) for class_id in predicted], probs, best_model.feature_importances_

BOOST_ROUNDS = 50000  # we use early stopping so make this arbitrarily high
RANDOMSTATE = 42


def cv_over_param_dict(X_train, y_train, param_dict, kfolds, verbose=False, description="Cross validating"):
    start_time = datetime.now()
    print("%-20s %s" % ("Start Time", start_time), len(param_dict))

    results = []

    for i, d in tqdm(enumerate(param_dict), desc=description):
        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=BOOST_ROUNDS,
            random_state=RANDOMSTATE,
            verbosity=1,
            n_jobs=-1,
            booster='gbtree',
            **d
        )

        scores = cross_val_score(clf, X_train, y_train, cv=kfolds,
                                 scoring=make_scorer(f1_score, greater_is_better=True))
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results.append([mean_score, std_score, d])

        print(
            "%s %3d result mean: %.6f std: %.6f" % (datetime.strftime(datetime.now(), "%T"), i, mean_score, std_score))

    end_time = datetime.now()
    print("%-20s %s" % ("Start Time", start_time))
    print("%-20s %s" % ("End Time", end_time))
    print(str((end_time - start_time).seconds))

    results_df = pd.DataFrame(results, columns=['f1_score', 'std', 'param_dict']).sort_values('f1_score',
                                                                                              ascending=False)
    display(results_df.head())

    best_params = results_df.iloc[0]['param_dict']
    return best_params, results_df


def train_and_predict_using_grid_search_xgboost(X_train, X_validation, y_train):
    """Tune hyperparameters for xgboost.

    Takes long. Script adapted from
    https://towardsdatascience.com/beyond-grid-search-hypercharge-hyperparameter-tuning-for-xgboost-7c78f7a2929d.
    """

    # convert y_train to numbers
    y_train_xgboost = [0 if v == 'moral disgust' else 1 for v in y_train]

    # initial hyperparams
    current_params = {
        'max_depth': 5,
        'colsample_bytree': 0.5,
        'colsample_bylevel': 0.5,
        'subsample': 0.5,
        'learning_rate': 0.01,
    }

    # number of folds for cross validation
    kfolds = 3

    # round 1: tune depth
    max_depths = list(range(2, 4))
    grid_search_dicts = [{'max_depth': md} for md in max_depths]
    # merge into full param dicts
    full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]
    # cv and get best params
    current_params, results_df = cv_over_param_dict(X_train, y_train_xgboost, full_search_dicts, kfolds,
                                                    description='Tune depth')

    # round 2: tune subsample, colsample_bytree, colsample_bylevel
    subsamples = np.linspace(0.25, 0.75, 5)
    colsample_bytrees = np.linspace(0.1, 0.3, 3)
    colsample_bylevel = np.linspace(0.1, 0.3, 3)

    grid_search_dicts = [dict(zip(['subsample', 'colsample_bytree', 'colsample_bylevel'], [a, b, c]))
                         for a, b, c in product(subsamples, colsample_bytrees, colsample_bylevel)]
    # merge into full param dicts
    full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]
    # cv and get best params
    current_params, results_df = cv_over_param_dict(X_train, y_train_xgboost, full_search_dicts, kfolds,
                                                    description='Tune subsample, colsample_bytree, colsample_bylevel')

    # round 3: learning rate
    learning_rates = np.logspace(-3, -1, 5)
    grid_search_dicts = [{'learning_rate': lr} for lr in learning_rates]
    # merge into full param dicts
    full_search_dicts = [{**current_params, **d} for d in grid_search_dicts]
    # cv and get best params
    current_params, results_df = cv_over_param_dict(X_train, y_train_xgboost, full_search_dicts, kfolds,
                                                    description='Tune learning rate')

    print("Best parameters: ", current_params)

    clf = xgb.XGBClassifier(**current_params)

    y_train_xgboost = [0 if v == 'moral disgust' else 1 for v in y_train]
    clf.fit(X_train, y_train_xgboost)

    predicted = clf.predict(X_validation)
    probs = clf.predict_proba(X_validation)[:, 1]
    return ['moral disgust' if v == 0 else 'pathogen disgust' for v in predicted], probs


# def train_and_predict_using_grid_search_xgboost(X_train, X_validation, y_train):
#     param_grid = {
#         'max_depth': [100],  # [100, 200, 300, 500],  # number of trees
#         'subsample': [10, 20, 30, None],  # max number of levels in each decision tree
#         'colsample_bytree': [2, 5, 10],  # min number of samples required to split a node
#         'colsample_bylevel': [4, 2, 4],  # min number of samples required at each leaf node
#         'learning_rate': [True, False]  # method for sampling data points (with or without replacement)
#         'n_estimators': [50, 100, 200]  # method for sampling data points (with or without replacement)
#     }
#
#     # Initialize the base classifier
#     clf = xgb.XGBClassifier()
#
#     # Initialize the grid search object
#
#     grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=3)
#
#     # Fit the grid search object to the data
#     grid_search.fit(X_train, y_train)
#
#     # Get the optimal hyperparameters
#     best_params = grid_search.best_params_
#     clf = xgb.XGBClassifier(**best_params)
#
#     y_train_xgboost = [0 if v == 'moral disgust' else 1 for v in y_train]
#     clf.fit(X_train, y_train_xgboost)
#
#     predicted = clf.predict(X_validation)
#     probs = clf.predict_proba(X_validation)[:, 1]
#     return ['moral disgust' if v == 0 else 'pathogen disgust' for v in predicted], probs


def train_and_predict_using_grid_search_rf(X_train, X_validation, y_train):
    # Define the hyperparameters you want to tune
    param_grid = {
        'n_estimators': [100],  # [100, 200, 300, 500],  # number of trees
        'max_depth': [10, 20, 30, None],  # max number of levels in each decision tree
        'min_samples_split': [2, 5, 10],  # min number of samples required to split a node
        'min_samples_leaf': [4, 2, 4],  # min number of samples required at each leaf node
        'bootstrap': [True, False]  # method for sampling data points (with or without replacement)
    }

    # Initialize the base classifier
    clf = RandomForestClassifier()

    # Initialize the grid search object

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=3)

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
    probs = clf_best.predict_proba(X_validation)[:, 1]
    return predicted, probs
