from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from classify_video_with_videomae import infer
from utils import Video, load_videos, parse_arguments, get_path_config_from_args


def main():
    config = get_path_config_from_args()
    videos = load_videos(config)

    check_video_files(videos)
    videos_with_paths = [v for v in videos if v.path is not None]
    videos_to_encode = [v for v in videos_with_paths if not v.has_features()]
    save_encode(videos_to_encode, videos, config.features_csv)


def save_encode(videos_to_encode, videos, features_csv):
    failed_videos = []
    for video in tqdm(videos_to_encode):
        try:
            _, logits = infer(str(video.path))
            video.features = np.array(logits)
            save_features(videos, features_csv)
        except Exception as e:
            video.error = str(e)
            failed_videos.append(video)
    report_failures(failed_videos)


def save_features(videos, features_csv):
    output = pd.DataFrame([(video.id, video.features) for video in videos], columns=('VideoID', 'features'))
    output.to_csv(features_csv)


def report_failures(failed_videos: List[Video]):
    if failed_videos:
        print(f'There were errors during loading or inference of the following {len(failed_videos)} video paths:')
        print(f'{",".join([str(video.path) for video in failed_videos])}')
        for video in failed_videos:
            print(f'For video {video.id} the following error occured: {video.error}')


def check_video_files(videos):
    videos_without_files = [v for v in videos if v.path is None]
    videos_with_files = [v for v in videos if v.path is not None]
    formatted_id_list = ",".join([str(v.id) for v in videos_without_files])
    print(
        f'Found {len(videos_with_files)} video files from all {len(videos)} videos in csv. Videos files for the following {len(videos_without_files)} videos are missing:\n{formatted_id_list}')


if __name__ == '__main__':
    main()
