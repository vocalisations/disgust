from pathlib import Path

import pandas as pd

from disgust.classify_video_with_videomae import infer


def test_classification():
    """Tests the accuracy of the classification inference for a specific video file."""
    expected_labels = ['skateboarding', 'roller skating']
    s = str(Path('./testdata') / 'videos' / '1.mp4')

    confidences, logits = infer(s)
    sorted_confidences = pd.DataFrame(sorted(confidences, key=lambda x: x[1], reverse=True),
                                      columns=('label', 'confidence'))

    print(sorted_confidences)
    actual_labels = sorted_confidences['label'].iloc[:2].tolist()
    assert actual_labels == expected_labels, f"Error: expected {expected_labels}, but got {actual_labels}"
