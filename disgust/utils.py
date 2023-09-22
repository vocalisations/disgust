import argparse
import json
import re
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from disgust.learners import available_learners
from disgust.learners.available_learners import available_learners
from disgust.models.available_models import available_models


@dataclass
class Video:
    id: str
    path: Optional[Path] = None
    features: Optional[np.ndarray] = None
    error: Optional[str] = None
    link: Optional[str] = None

    def has_features(self) -> bool:
        return isinstance(self.features, np.ndarray)


def get_features_csv_path(model: str, meta_csv: Path):
    if model not in available_models:
        raise ValueError(f'Invalid model "{model}" selected; choose from {list(available_models.keys())}')
    features_csv = meta_csv.parent / f"{meta_csv.stem}_{model}_logits.csv"
    return features_csv


def load_videos(meta_csv_path, model_type, video_dir):
    videos = [Video(video_id, path=get_video_path(video_dir, video_id), link=video_link) for video_id, video_link in
              read_video_ids_and_links(meta_csv_path)]

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


def read_video_ids_and_links(meta_csv_path: Path):
    csv = pd.read_csv(meta_csv_path)
    return zip(csv['VideoID'].astype(str), csv['Link'].astype(str))


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


def create_split_videos_masks(links, r_train=0.67, r_validation=0.16):
    """Clusters links by their id with counts and then divides the links over 3 splits, train, validation and test as
    close the given ratios as possible, making sure that only complete clusters are assigned to each split."""
    # Create clusters
    clusters = {}
    for i_link, link in enumerate(links):
        clusters.setdefault(get_id_from_video_link(link), []).append(i_link)

    # Sort clusters by size
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    total_count = len(links)
    train_count, validation_count = int(r_train * total_count), int(r_validation * total_count)

    train_indices, validation_indices, test_indices = [], [], []
    for cluster in sorted_clusters:
        if len(train_indices) + len(cluster) <= train_count:
            train_indices.extend(cluster)
        elif len(validation_indices) + len(cluster) <= validation_count:
            validation_indices.extend(cluster)
        else:
            test_indices.extend(cluster)

    return train_indices, validation_indices, test_indices


def get_id_from_video_link(video_link):
    if 'vm.tiktok.com' in video_link:
        match = re.search(r'https://vm\.tiktok\.com/(.+?)/', video_link)
        if match:
            return 'tt' + match.group(1)

    if 'youtube.com' in video_link:
        match = re.search(r'v=([^&]+)', video_link)
        if match:
            return 'yt' + match.group(1)

    if 'youtu.be' in video_link:
        match = re.search(r'https://youtu\.be/([^/]+)', video_link)
        if match:
            return 'yt' + match.group(1)
