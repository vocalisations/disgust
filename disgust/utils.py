import argparse
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from krippendorff import krippendorff
from pandas.io.formats.style import Styler
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from pandas import DataFrame as df, DataFrame

from disgust.learners import available_learners
from disgust.learners.available_learners import available_learners
from disgust.models.available_models import available_models
from matplotlib import pyplot as plt


@dataclass
class Video:
    id: str
    path: Optional[Path] = None
    features: Optional[np.ndarray] = None
    error: Optional[str] = None

    def has_features(self) -> bool:
        return isinstance(self.features, np.ndarray)


def get_features_csv_path(model: str, meta_csv: Path):
    if model not in available_models:
        raise ValueError(f'Invalid model "{model}" selected; choose from {list(available_models.keys())}')
    features_csv = meta_csv.parent / f"{meta_csv.stem}_{model}_logits.csv"
    return features_csv


def load_videos(meta_csv_path, model_type, video_dir):
    videos = [Video(video_id, path=get_video_path(video_dir, video_id)) for video_id in
              read_video_ids(meta_csv_path)]

    model_types = available_models.keys() if model_type == 'all' else [model_type]
    for current_model_type in model_types:
        features_csv = get_features_csv_path(current_model_type, meta_csv_path)
        copy_existing_features(videos, features_csv, current_model_type)
    return videos


def parse_arguments(requested_args):
    """Parse only the requested arguments from the command line."""
    parser = argparse.ArgumentParser()
    available_args = {
        'meta_csv': {'type': Path, 'help': 'Path to the csv file containing a column called VideoID.'},
        'video_dir': {'type': Path, 'help': 'Path to folder containing the video files.'},
        'model': {'type': str, 'help': f'model type; choose from {list(available_models.keys()) + ["all"]}.'},
        'learner_type': {'type': str, 'help': f'model type; choose from {list(available_learners.keys())}.'},
    }

    for requested_arg in requested_args:
        parser.add_argument(requested_arg, **available_args[requested_arg])
    args = parser.parse_args()
    return [getattr(args, r) for r in requested_args]


def read_video_ids(meta_csv_path: Path):
    return pd.read_csv(meta_csv_path)['VideoID'].astype(str)


def get_video_path(video_dir: Path, video_id: str) -> Optional[Video]:
    matches = [f for f in video_dir.iterdir() if f.stem == video_id]
    if not matches:
        return None
    return matches[0]


def read_feature_set(text: str):
    if text == 'nan' or isinstance(text, float):
        return None

    obj = None
    try:
        obj = np.array(json.loads(text))
    except (JSONDecodeError, TypeError) as e:
        print('Error while decoding the following json features:', text, 'with type', type(text))
    if not isinstance(obj, np.ndarray):
        return None
    if np.isnan(obj).any():
        return None
    return obj


def copy_existing_features(videos: List[Video], features_csv: Path, model_type: str):
    existing_videos = [Video(str(t['VideoID']), features=read_feature_set(t['features'])) for _index, t in
                       pd.read_csv(features_csv).iterrows()] if features_csv.exists() else {}

    recovered_videos = []
    for video in videos:
        existing_video = get_matching_video(existing_videos, video.id)
        if existing_video is not None and existing_video.has_features():
            if video.has_features():
                video.features = np.concatenate((video.features, existing_video.features))
            else:
                video.features = existing_video.features
            recovered_videos.append(video)

    if recovered_videos:
        print(f'Loaded previously computed features of type "{model_type}" for {len(recovered_videos)} videos.')
    else:
        print(f'Found no previously computed features of type "{model_type}".')


def get_matching_video(existing_videos: List[Video], video_id: str):
    return next((existing_video for existing_video in existing_videos if existing_video.id == video_id), None)


def display_performance_metrics(trues, predicted, probs, class_list):
    class_metrics, general_metrics, roc, conf_matrix = calculate_performance_metrics(trues, predicted, probs,
                                                                                     class_list)

    formatted_tables = format_tables(class_metrics, general_metrics, conf_matrix)

    for table in formatted_tables:
        display(table)

    if roc is not None:
        display(roc)


def save_performance_metrics(trues, predicted, probs, class_list, folder: Path):
    class_metrics, general_metrics, roc, conf_matrix = calculate_performance_metrics(trues, predicted, probs,
                                                                                     class_list)

    class_metrics_f, general_metrics_f, conf_matrix_f = format_tables(class_metrics, general_metrics, conf_matrix)

    if roc is not None:
        roc_auc = general_metrics.loc['auc', 'score']
        plt.title('Receiver Operating Characteristic')
        plt.plot(roc['fpr'], roc['tpr'], 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(folder / 'roc.svg')

    doc = f'<div>{general_metrics_f.to_html()}</div><div>{conf_matrix_f.to_html()}</div>' \
          f'<div>{class_metrics_f.to_html()}</div><div><img src = "roc.svg" /></div>'
    with open(folder / 'performance_metrics.html', 'w') as f:
        f.write(doc)



def format_tables(class_metrics: DataFrame, general_metrics: DataFrame, conf_matrix: DataFrame) -> tuple[Styler]:
    formatted_class_metrics, formatted_general_metrics = [
        table.style.background_gradient(vmin=0, vmax=1, cmap='Greys_r').set_precision(2) for table in
        [class_metrics, general_metrics]]
    formatted_conf_matrix = conf_matrix.style.background_gradient(vmin=0, vmax=conf_matrix.sum().sum(),
                                                                  cmap='Greys_r').set_precision(
        2).format(lambda c: f'{c} ({100 * c / conf_matrix.sum().sum():.0f}%)')

    return formatted_class_metrics, formatted_general_metrics, formatted_conf_matrix


def print_performance_metrics(trues, predicted, probs, class_list):
    class_metrics, general_metrics, roc, conf_matrix = calculate_performance_metrics(trues, predicted, probs,
                                                                                     class_list)
    print(class_metrics.round(2))
    print(general_metrics.round(2))
    # if roc is not None:
    # print(roc)
    print(conf_matrix)


def calculate_performance_metrics(trues, predicted, probs, class_list):
    """
    Calculates some performance metrics given a list of ground truth values and a list of predictions to be compared.
    :param trues: list of ground truths
    :param predicted: list of model predictions
    :param probs: list of model predicted probalities
    :param class_list: the set of all possible labels
    :return: a dataframe with class level metrics and a dataframe with general metrics and a one with ROC values
    """
    class_metrics_data = {'recall': recall_score(trues, predicted, average=None),
                          'precision': precision_score(trues, predicted, average=None),
                          'f1': f1_score(trues, predicted, average=None)}
    class_metrics = df(class_metrics_data, index=class_list)

    i_trues = [list(class_list).index(label) for label in trues]
    i_predicted = [list(class_list).index(label) for label in predicted]
    i_set = np.unique(i_trues + i_predicted)

    if probs is not None:
        from sklearn.metrics import roc_auc_score, roc_curve
        fpr, tpr, thresholds = roc_curve(y_true=trues, y_score=probs, pos_label='pathogen disgust')
        roc = df({'fpr': fpr, 'tpr': tpr})
        roc_auc = (roc_auc_score(trues, probs))
    else:
        roc = None
        roc_auc = None

    general_metrics_data = [roc_auc,
                            accuracy_score(trues, predicted),
                            krippendorff.alpha(reliability_data=[i_trues, i_predicted],
                                               level_of_measurement='nominal', value_domain=i_set)]
    general_metrics = df(general_metrics_data, index=['auc', 'accuracy', 'krippendorff alpha'], columns=['score'])

    conf_matrix = pd.DataFrame(confusion_matrix(trues, predicted),
                               index=pd.MultiIndex.from_product([['True:'], class_list]),
                               columns=pd.MultiIndex.from_product([['Predicted:'], class_list]))

    return class_metrics, general_metrics[general_metrics['score'].notna()], roc, conf_matrix
