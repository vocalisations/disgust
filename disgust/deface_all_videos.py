import argparse
import os
import subprocess
import sys
from pathlib import Path

from typing import Tuple

from tqdm import tqdm

DESCRIPTION = f"This script takes a folder path containing video files and uses deface to automatically blur any faces from humans in them."
EPILOGUE = f"Example usage: python {Path(__file__).name} <source_folder_path> <output_folder_path>"


def parse_arguments() -> Tuple[Path, Path]:
    parser = argparse.ArgumentParser()
    parser.epilog = EPILOGUE
    parser.description = DESCRIPTION
    parser.add_argument('source_folder_path', type=Path, help=f"path to directory containing video files")
    parser.add_argument('target_folder_path', type=Path,
                        help=f"path to output directory that will contain anonymized video files")
    args = parser.parse_args(sys.argv[1:])
    source_folder_path = args.source_folder_path
    target_folder_path = args.target_folder_path
    return source_folder_path, target_folder_path


def deface_folder(source_folder_path: Path, target_folder_path: Path):
    if not source_folder_path.exists():
        raise SystemExit(f"Error: Invalid argument '{source_folder_path}' - must be a valid file path.")
    os.makedirs(target_folder_path, exist_ok=True)
    video_file_extensions = ['.avi', '.mp4']
    source_file_paths = [p for p in source_folder_path.glob("*") if p.is_file() and p.suffix in video_file_extensions and '_an']
    for source_file_path in tqdm(source_file_paths):
        target_file_path = target_folder_path / source_file_path.name
        deface_video(source_file_path, target_file_path)


def deface_video(input_file, output_file):
    command = ['deface', input_file, '--output', output_file, '--keep-audio']
    subprocess.run(command, check=True)


def main():
    source_folder_path, target_folder_path = parse_arguments()
    deface_folder(source_folder_path, target_folder_path)


if __name__ == '__main__':
    main()
