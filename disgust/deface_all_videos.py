import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

from deface import deface
from deface.centerface import CenterFace
from tqdm import tqdm

DESCRIPTION = f"This script takes a folder path containing video files and uses deface to automatically blur any human faces in them."
EPILOGUE = f"Example usage: python {Path(__file__).name} <source_folder_path> <output_folder_path>"


def parse_arguments() -> Tuple[Path, Path]:
    parser = argparse.ArgumentParser()
    parser.epilog = EPILOGUE
    parser.description = DESCRIPTION
    parser.add_argument('source_folder_path', type=Path, help=f"path to directory containing video files")
    parser.add_argument('target_folder_path', type=Path,
                        help=f"path to output directory that will contain anonymized video files")
    args = parser.parse_args(sys.argv[1:])
    return args.source_folder_path, args.target_folder_path


def deface_folder(source_folder_path: Path, target_folder_path: Path):
    if not source_folder_path.exists():
        raise SystemExit(f"Error: Invalid argument '{source_folder_path}' - must be a valid file path.")
    os.makedirs(target_folder_path, exist_ok=True)
    video_file_extensions = ['.avi', '.mp4']
    source_file_paths = [p for p in source_folder_path.glob("*") if p.is_file() and p.suffix in video_file_extensions and '_an']
    for source_file_path in tqdm(source_file_paths, desc='Defacing videos'):
        target_file_path = target_folder_path / source_file_path.name
        deface_video(source_file_path, target_file_path)


def deface_video(input_file, output_file):
    """Call defacing tool to blur faces using default values."""
    deface.video_detect(str(input_file),
                        str(output_file),
                        centerface=CenterFace(),
                        threshold=0.2,
                        enable_preview=None,
                        cam=None,
                        nested=None,
                        replacewith=None,
                        mask_scale=1.3,
                        ellipse=None,
                        draw_scores=None,
                        ffmpeg_config={"codec": "libx264"},
                        replaceimg=None,
                        keep_audio=True, )


def main():
    source_folder_path, target_folder_path = parse_arguments()
    deface_folder(source_folder_path, target_folder_path)


if __name__ == '__main__':
    main()
