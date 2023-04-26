import argparse
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils import load_videos, parse_arguments, get_path_config_from_args, print_performance_metrics
import pandas as pd


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
    n_trees = 200
    clf = RandomForestClassifier(n_estimators=n_trees)


    clf.fit(X_train, y_train)

    predicted = clf.predict(X_validation)
    print_performance_metrics(trues=y_validation, predicted=predicted, class_list=y_train.unique())
    print(confusion_matrix(y_validation, predicted), 'true pathogen disgust:', len([p for p in predicted if p == 'pathogen disgust']))



if __name__ == '__main__':
    main()
