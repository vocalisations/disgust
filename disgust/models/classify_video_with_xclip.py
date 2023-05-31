import argparse
from pathlib import Path
from typing import List

import cv2
import imutils
import numpy as np
import pandas as pd
import torch
from torchvision.io.video import av
from transformers import AutoModel, AutoProcessor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=Path, help='Path to the video file.')
    return parser.parse_args()



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


def read_video_pyav(container: av.container.input.InputContainer, indices: List[int]):
    """Decode the video with PyAV decoder.

        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        :param container: PyAV container
        :param indices: List of frame indices to decode
        :return:

    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def infer(video_file: str, return_classifications=False, return_logits=True):
    container = av.open(video_file)

    # sample 8 frames
    np.random.seed(0)
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
    inputs = processor(videos=list(video), return_tensors="pt")

    if return_logits:
        video_features = model.get_video_features(**inputs).detach()
        print(video_features.shape)
    else:
        video_features = []

    if return_classifications:
        classes = ["playing sports", "eating spaghetti", "garbage bag", "go shopping", "inline skating", "watching tv"]
        probs = classify(classes, model, processor, video)
    else:
        probs = []

    return probs, video_features


def classify(classes, model, processor, video):
    inputs = processor(

        text=classes,

        videos=list(video),

        return_tensors="pt",

        padding=True,

    )
    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
    probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return [(label, float(prob)) for label, prob in zip(classes, probs.tolist()[0])]


def main():
    args = parse_arguments()
    video_path = args.video_path
    confidences, _ = infer(str(video_path))
    sorted_confidences = pd.DataFrame(sorted(confidences, key=lambda x: x[1], reverse=True),
                                      columns=('label', 'confidence'))

    print(sorted_confidences)


if __name__ == '__main__':
    main()
