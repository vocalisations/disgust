import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_csv', type=Path, help='Path to the csv file containing a column called VideoID.')
    parser.add_argument('video_dir', type=Path, help='Path to folder containing the video files.')
    parser.add_argument('--output_csv', type=Path, help='Path to the csv file containing a column called VideoID.',
                        required=False)
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
    video_path: Path
    videomae_features: np.array


def main():
    args = parse_arguments()
    meta_csv_path = args.meta_csv
    video_dir = args.video_dir
    output_csv = args.output_csv
    if output_csv is None:
        file_name = meta_csv_path.stem + '.videomae_logits.csv'
        output_csv = meta_csv_path.parent / file_name

    video_ids = read_video_ids(meta_csv_path)
    videos = [Video(video_id, get_video_path(video_dir, video_id), None) for video_id in video_ids]

    videos_without_files = [v for v in videos if v.video_path is None]

    formatted_id_list = "\n".join([str(v.id) for v in videos_without_files])
    print(f'The following {len(videos_without_files)} of all {len(videos)} videos are missing:\n{formatted_id_list}')



if __name__ == '__main__':
    main()
