import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Video:
    id: str
    path: Optional[Path] = None
    features: Optional[np.array] = None
    error: Optional[str] = None

    def has_features(self) -> bool:
        print(self.features, type(self.features))
        return isinstance(self.features, np.ndarray)


def load_videos(args):
    meta_csv_path = args.meta_csv
    video_dir = args.video_dir
    features_csv = args.features_csv if args.features_csv else \
        meta_csv_path.parent / (meta_csv_path.stem + '.videomae_logits.csv')
    videos = [Video(video_id, path=get_video_path(video_dir, video_id)) for video_id in read_video_ids(meta_csv_path)]
    copy_existing_features(videos, features_csv)
    return videos, features_csv


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_csv', type=Path, help='Path to the csv file containing a column called VideoID.')
    parser.add_argument('video_dir', type=Path, help='Path to folder containing the video files.')
    parser.add_argument('--features_csv', type=Path, help='Path to the csv file containing a column called VideoID.')
    return parser.parse_args()


def read_video_ids(meta_csv_path):
    return pd.read_csv(meta_csv_path)['VideoID'].astype(str)


def get_video_path(video_dir: Path, video_id: str):
    matches = [f for f in video_dir.iterdir() if f.stem == video_id]
    if not matches:
        return None
    return matches[0]


def copy_existing_features(videos, features_csv):
    existing_videos = [Video(str(t['VideoID']), features=t['features']) for _index, t in
                       pd.read_csv(features_csv).iterrows()] if features_csv.exists() else {}

    recovered_videos = []
    for video in videos:
        existing_video = get_matching_video(existing_videos, video.id)
        if existing_video is not None and existing_video.has_features():
            video.features = existing_video.features
            recovered_videos.append(video)

    if recovered_videos:
        print(f'Recovered previously computed features for {len(recovered_videos)} videos.')
    else:
        print(f'Found no previously computed features so we will start from 0 and compute all videos.')


def get_matching_video(existing_videos, video_id):
    return next((existing_video for existing_video in existing_videos if existing_video.id == video_id), None)
