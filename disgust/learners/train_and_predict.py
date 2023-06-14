import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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


