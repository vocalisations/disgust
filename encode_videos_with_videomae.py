import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from classify_video_with_videomae import infer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_csv', type=Path, help='Path to the csv file containing a column called VideoID.')
    parser.add_argument('video_dir', type=Path, help='Path to folder containing the video files.')
    parser.add_argument('--output_csv', type=Path, help='Path to the csv file containing a column called VideoID.')
    return parser.parse_args()


def read_video_ids(meta_csv_path):
    return pd.read_csv(meta_csv_path)['VideoID']


def get_video_path(video_dir: Path, video_id: str):
    matches = [f for f in video_dir.iterdir() if f.stem == video_id]
    if not matches:
        return None
    return matches[0]


@dataclass
class Video:
    id: str
    path: Optional[Path] = None
    features: Optional[np.array] = None
    error: Optional[str] = None


def main():
    args = parse_arguments()
    meta_csv_path = args.meta_csv
    video_dir = args.video_dir

    output_csv = args.output_csv if args.output_csv else \
        meta_csv_path.parent / (meta_csv_path.stem + '.videomae_logits.csv')

    videos = [Video(video_id, path=get_video_path(video_dir, video_id)) for video_id in read_video_ids(meta_csv_path)]

    check_video_files(videos)
    copy_existing_features(videos, output_csv)
    videos_with_paths = [v for v in videos if v.path is not None]
    videos_to_encode = [v for v in videos_with_paths if v.features is None]

    save_encode(videos_to_encode, videos, output_csv)


def save_encode(videos_to_encode, videos, output_csv):
    failed_videos = []
    for video in tqdm(videos_to_encode):
        try:
            _, logits = infer(str(video.path))
            video.features = logits
            save_features(videos, output_csv)
        except Exception as e:
            video.error = str(e)
            failed_videos.append(video)
    report_failures(failed_videos)


def report_failures(failed_videos):
    if failed_videos:
        print(f'There were errors during loading or inference of the following {len(failed_videos)} video paths:')
        print(f'{",".join([video.path for video in failed_videos])}')
        for video in failed_videos:
            print(f'For video {video.id} the following error occured: {video.error}')


def copy_existing_features(videos, output_csv):
    existing_videos = [Video(t['VideoID'], features=t['features']) for _index, t in
                       pd.read_csv(output_csv).iterrows()] if output_csv.exists() else {}

    recovered_videos = []
    for video in videos:
        existing_video = get_matching_video(existing_videos, video.id)
        if existing_video is not None and existing_video.features is not None:
            video.features = existing_video.features
            recovered_videos.append(video)

    if recovered_videos:
        print(f'Recovered previously computed features for {len(recovered_videos)} videos.')
    else:
        print(f'Found no previously computed features so we will start from 0 and compute all videos.')


def get_matching_video(existing_videos, video_id):
    return next((existing_video for existing_video in existing_videos if existing_video.id is video_id), None)


def check_video_files(videos):
    videos_without_files = [v for v in videos if v.path is None]
    videos_with_files = [v for v in videos if v.path is not None]
    formatted_id_list = ",".join([str(v.id) for v in videos_without_files])
    print(
        f'Found {len(videos_with_files)} video files from all {len(videos)} videos in csv. Videos files for the following {len(videos_without_files)} videos are missing:\n{formatted_id_list}')


def save_features(videos, output_csv):
    output = pd.DataFrame([(video.id, video.features) for video in videos], columns=('VideoID', 'features'))
    output.to_csv(output_csv)


if __name__ == '__main__':
    main()