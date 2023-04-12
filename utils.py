import argparse
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass
class Video:
    id: str
    path: Optional[Path] = None
    features: Optional[np.ndarray] = None
    error: Optional[str] = None

    def has_features(self) -> bool:
        return isinstance(self.features, np.ndarray)


@dataclass
class PathConfig:
    meta_csv_path: Path
    video_dir: Path
    features_csv: Path


def get_path_config_from_args() -> PathConfig:
    args = parse_arguments()
    features_csv = args.features_csv if args.features_csv else \
        args.meta_csv.parent / (args.meta_csv.stem + '.videomae_logits.csv')
    return PathConfig(args.meta_csv, args.video_dir, features_csv)


def load_videos(config: PathConfig):
    videos = [Video(video_id, path=get_video_path(config.video_dir, video_id)) for video_id in
              read_video_ids(config.meta_csv_path)]
    copy_existing_features(videos, config.features_csv)
    return videos


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_csv', type=Path, help='Path to the csv file containing a column called VideoID.')
    parser.add_argument('video_dir', type=Path, help='Path to folder containing the video files.')
    parser.add_argument('--features_csv', type=Path, help='Path to the csv file containing a column called VideoID.')
    return parser.parse_args()


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


def copy_existing_features(videos: List[Video], features_csv: Path):
    existing_videos = [Video(str(t['VideoID']), features=read_feature_set(t['features'])) for _index, t in
                       pd.read_csv(features_csv).iterrows()] if features_csv.exists() else {}

    recovered_videos = []
    for video in videos:
        existing_video = get_matching_video(existing_videos, video.id)
        if existing_video is not None and existing_video.has_features():
            video.features = existing_video.features
            recovered_videos.append(video)

    if recovered_videos:
        print(f'Loaded previously computed features for {len(recovered_videos)} videos.')
    else:
        print(f'Found no previously computed features.')


def get_matching_video(existing_videos: List[Video], video_id: str):
    return next((existing_video for existing_video in existing_videos if existing_video.id == video_id), None)
