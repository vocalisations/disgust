import argparse
from pathlib import Path

import cv2
import imutils
import numpy as np
import pandas as pd
import torch
from pytorchvideo.transforms import (
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    Resize,
)
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification

parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=Path, help='Path to the video file.')
args = parser.parse_args()
video_path = args.video_path
print(video_path)

MODEL_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = VideoMAEForVideoClassification.from_pretrained(MODEL_CKPT).to(DEVICE)
PROCESSOR = VideoMAEFeatureExtractor.from_pretrained(MODEL_CKPT)

RESIZE_TO = PROCESSOR.size["shortest_edge"]
NUM_FRAMES_TO_SAMPLE = MODEL.config.num_frames
IMAGE_STATS = {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}
VAL_TRANSFORMS = Compose(
    [
        UniformTemporalSubsample(NUM_FRAMES_TO_SAMPLE),
        Lambda(lambda x: x / 255.0),
        Normalize(IMAGE_STATS["image_mean"], IMAGE_STATS["image_std"]),
        Resize((RESIZE_TO, RESIZE_TO)),
    ]
)
LABELS = list(MODEL.config.label2id.keys())

def parse_video(video_file):
    """A utility to parse the input videos.
    Reference: https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
    """
    vs = cv2.VideoCapture(video_file)

    # try to determine the total number of frames in the video file
    try:
        prop = (
            cv2.cv.CV_CAP_PROP_FRAME_COUNT
            if imutils.is_cv2()
            else cv2.CAP_PROP_FRAME_COUNT
        )
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    frames = []

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

    return frames


def preprocess_video(frames: list):
    """Utility to apply preprocessing transformations to a video tensor."""
    # Each frame in the `frames` list has the shape: (height, width, num_channels).
    # Collated together the `frames` has the the shape: (num_frames, height, width, num_channels).
    # So, after converting the `frames` list to a torch tensor, we permute the shape
    # such that it becomes (num_channels, num_frames, height, width) to make
    # the shape compatible with the preprocessing transformations. After applying the
    # preprocessing chain, we permute the shape to (num_frames, num_channels, height, width)
    # to make it compatible with the model. Finally, we add a batch dimension so that our video
    # classification model can operate on it.
    video_tensor = torch.tensor(np.array(frames).astype(frames[0].dtype))
    video_tensor = video_tensor.permute(
        3, 0, 1, 2
    )  # (num_channels, num_frames, height, width)
    video_tensor_pp = VAL_TRANSFORMS(video_tensor)
    video_tensor_pp = video_tensor_pp.permute(
        1, 0, 2, 3
    )  # (num_frames, num_channels, height, width)
    video_tensor_pp = video_tensor_pp.unsqueeze(0)
    return video_tensor_pp.to(DEVICE)


def infer(video_file):
    frames = parse_video(video_file)
    video_tensor = preprocess_video(frames)
    inputs = {"pixel_values": video_tensor}

    # forward pass
    with torch.no_grad():
        outputs = MODEL(**inputs)
        logits = outputs.logits
    softmax_scores = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
    confidences = [(LABELS[i], float(softmax_scores[i])) for i in range(len(LABELS))]
    return confidences

confidences = infer(str(video_path))
sorted_confidences = pd.DataFrame(sorted(confidences, key=lambda x: x[1], reverse=True), columns=('label', 'confidence'))

print(sorted_confidences)

