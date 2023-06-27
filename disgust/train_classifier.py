import random

import numpy as np
import pandas as pd
from pretty_confusion_matrix import pp_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from disgust import disgust_classes
from disgust.classify_video import get_feature_names
from disgust.learners.available_learners import available_learners
from disgust.utils import parse_arguments
from utils import load_videos, print_performance_metrics


def main(use_pretty_confusion_matrix=True):
    meta_csv_path, video_dir, model, learner_type = parse_arguments(
        requested_args=['meta_csv', 'video_dir', 'model', 'learner_type'])

    videos = load_videos(meta_csv_path, model, video_dir)

    videos_with_features = [v for v in videos if v.has_features()]

    print(f'Loaded {len(videos_with_features)} videos with features out of a total of {len(videos)} videos.')

    X = np.stack([v.features for v in videos_with_features])

    ids = [v.id for v in videos_with_features]
    metadata = pd.read_csv(meta_csv_path)
    filtered_metadata = metadata[metadata['VideoID'].isin(ids)]
    sorted_metadata = filtered_metadata.sort_values(by=['VideoID'],
                                                    key=lambda x: pd.Categorical(x, categories=ids, ordered=True))
    y = sorted_metadata['Disgust category']

    X_train, X_validation, y_train, y_validation, X_test, y_test = split_dataset(X, y)

    predicted_classes, probabilities, feature_importances = train_and_predict(X_train, X_validation, y_train,
                                                                              learner_type=learner_type)
    evaluate(predicted_classes,
             probabilities,
             feature_importances,
             use_pretty_confusion_matrix,
             y_train,
             y_validation,
             get_feature_names(model))


def evaluate(predicted_classes, probabilities, feature_importances, use_pretty_confusion_matrix, y_train, y_validation,
             feature_names):
    feature_importances = pd.DataFrame(feature_importances, index=feature_names,
                                       columns=['importance']).sort_values(
        'importance', ascending=False)
    feature_importances['cumulative'] = feature_importances['importance'].cumsum()
    print(feature_importances[:30])
    print_performance_metrics(trues=y_validation, predicted=predicted_classes, probs=probabilities,
                              class_list=y_train.unique())
    conf_matrix = confusion_matrix(y_validation, predicted_classes)
    print(conf_matrix, 'true pathogen disgust:',
          len([p for p in predicted_classes if p == 'pathogen disgust']))
    # get pandas dataframe
    df_cm = pd.DataFrame(conf_matrix, index=(disgust_classes.class_names), columns=(disgust_classes.class_names))
    # colormap: see this and choose your more dear
    cmap = 'PuRd'
    if use_pretty_confusion_matrix:
        pp_matrix(df_cm, cmap=cmap, fmt='.1f', fz=11, figsize=[4, 4])


def split_dataset(X, y):
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.33, random_state=0)
    X_validation, X_test, y_validation, y_test = train_test_split(X_rest, y_rest, test_size=0.50, random_state=0)
    print(f'Split {len(y)} samples into the following sets:')
    print(f'Train set {len(y_train)}')
    print(f'Validation set {len(y_validation)}')
    print(f'Test set {len(y_test)}')
    return X_train, X_validation, y_train, y_validation, X_test, y_test


def train_and_predict(X_train, X_validation, y_train, learner_type: str):
    print(f'Training and predicting using {learner_type}.')

    if learner_type not in available_learners:
        raise ValueError(f'Invalid learner type "{learner_type}" selected. Choose from {available_learners.keys()}')

    return available_learners[learner_type](X_train, X_validation, y_train)


if __name__ == '__main__':
    main()
