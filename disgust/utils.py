import argparse
import json
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



