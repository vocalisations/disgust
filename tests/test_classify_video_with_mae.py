import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from disgust.classify_video import infer


def test_classification_videomae(test_video_1_path):
    """Tests the accuracy of the classification inference for a specific video file using videomae."""
    expected_labels = ['skateboarding', 'roller skating']
    model = 'videomae'

    assert_top2_expected_classifications(expected_labels, model, test_video_1_path)


def test_inference_with_xclip(random_seed, test_video_1_path):
    """Tests the accuracy of the classification inference for a specific video file using xclip."""
    expected_labels = ['inline skating', 'playing sports']
    model = 'xclip'

    assert_top2_expected_classifications(expected_labels, model, test_video_1_path)


def assert_top2_expected_classifications(expected_labels, model, test_video_1_path):
    confidences, logits = infer(test_video_1_path, return_classifications=True, model=model)
    sorted_confidences = pd.DataFrame(sorted(confidences, key=lambda x: x[1], reverse=True),
                                      columns=('label', 'confidence'))
    print(sorted_confidences)
    actual_labels = sorted_confidences['label'].iloc[:2].tolist()
    assert actual_labels == expected_labels, f"Error: expected {expected_labels}, but got {actual_labels}"


@pytest.fixture
def random_seed():
    """Set the random seed."""
    np.random.seed(0)
    random.seed(0)

@pytest.fixture
def test_video_1_path():
    """Set the random seed."""
    return str(Path(__file__).resolve().parent / 'testdata' / 'videos' / '1.mp4')

