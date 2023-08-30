import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from disgust.classify_video import infer
from disgust.utils import Video, load_videos, parse_arguments, \
    get_features_csv_path


def main():
    meta_csv_path, video_dir, model_type = parse_arguments(requested_args=['meta_csv', 'video_dir', 'model'])
    features_csv = get_features_csv_path(model_type, meta_csv_path)

    videos = load_videos(meta_csv_path, model_type, video_dir)

    videos_with_paths = [v for v in videos if v.path is not None]
    videos_to_encode = [v for v in videos_with_paths if not v.has_features()]
    check_video_files(videos_to_encode)
    save_encode(videos_to_encode, videos, features_csv, model_type)


def save_encode(videos_to_encode, videos, features_csv, model):
    failed_videos = []
    for video in tqdm(videos_to_encode):
        try:
            _, logits = infer(str(video.path), model=model, return_logits=True, return_classifications=False)
        except Exception as e:
            video.error = str(e)
            failed_videos.append(video)
            continue

        video.features = np.array(logits[0].cpu())  # Remove batch dimension
        save_features(videos, features_csv)

    report_failures(failed_videos)


def save_features(videos: List[Video], features_csv: Path):
    def serialize_features(features):
        return json.dumps(features.tolist() if isinstance(features, np.ndarray) else None)

    output = pd.DataFrame([(video.id, serialize_features(video.features)) for video in videos],
                          columns=('VideoID', 'features'))
    output.to_csv(features_csv)


def report_failures(failed_videos: List[Video]):
    if failed_videos:
        print(f'There were errors during loading or inference of the following {len(failed_videos)} video paths:')
        print(f'{",".join([str(video.path) for video in failed_videos])}')
        for video in failed_videos:
            print(f'For video {video.id} the following error occured: {video.error}')


def check_video_files(videos):
    if not videos:
        return

    videos_without_files = [v for v in videos if v.path is None]
    videos_with_files = [v for v in videos if v.path is not None]
    formatted_id_list = ",".join([str(v.id) for v in videos_without_files])
    print(f'Found {len(videos_with_files)} video files from all {len(videos)} videos to encode.')
    if videos_without_files:
        print(f'Videos files for the following {len(videos_without_files)} videos are missing:\n{formatted_id_list}')


if __name__ == '__main__':
    main()
