from pathlib import Path

import pandas as pd

from classify_video_with_videomae import infer


def test_classification():
    s = str(Path('testdata') / 'videos' / '1.mp4')
    confidences, logits = infer(s)
    sorted_confidences = pd.DataFrame(sorted(confidences, key=lambda x: x[1], reverse=True),
                                      columns=('label', 'confidence'))
    print(sorted_confidences)
