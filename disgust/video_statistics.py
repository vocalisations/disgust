from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import cv2 as cv
import argparse

from tqdm import tqdm


def get_duration(video_path: Path) -> float:
    """Read video duration (seconds) from a video file."""
    cap = cv.VideoCapture(video_path)
    frames_per_second = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / frames_per_second


def get_video_durations(folder_path: Path) -> list[float]:
    """Read all videos in a folder an return a list of video durations in seconds."""
    video_durations = []
    for filename in tqdm(os.listdir(folder_path), desc='Reading video durations'):
        if filename.endswith('.mp4') or filename.endswith('.avi'):  # Update with the desired video file extensions
            video_path = os.path.join(folder_path, filename)
            duration = get_duration(video_path)
            video_durations.append(duration)
    return video_durations


def plot_histogram(data: list[float],
                   num_bins: int = 200,
                   max_value: Optional[float] = None,
                   output_figure_path: Optional[Path] = None,
                   ):
    """Plots a histogram of a list of video durations."""
    min_value = min(data)
    max_value = max_value if max_value else max(data)
    bin_width = (max_value - min_value) / num_bins

    hist, bins = np.histogram(data, bins=num_bins, range=(min_value, max_value))

    plt.bar(bins[:-1], hist, width=bin_width, align='edge')
    plt.title('Video duration distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of occurrences')

    if output_figure_path is not None:
        plt.savefig(output_figure_path)
    else:
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process video files in a specified folder.')
    parser.add_argument('folder_path', type=Path, help='Path to the folder containing the video files.')
    parser.add_argument('--output_figure_path', type=Path, help='Path to which to write a plot to.')
    args = parser.parse_args()
    output_figure_path = args.output_figure_path if args.output_figure_path else args.folder_path / '_durations.png'
    folder_path = args.folder_path
    return folder_path, output_figure_path


def main():
    folder_path, output_figure_path = parse_arguments()
    durations = get_video_durations(folder_path)
    plot_histogram(durations, output_figure_path=output_figure_path)


if __name__ == '__main__':
    main()
