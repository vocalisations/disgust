import json
from pathlib import Path

from disgust.utils import save_performance_metrics


def test_save_performance_metrics(tmpdir):
    """4 files should be written to."""
    with open('testdata/trues_preds_probs_classnames.json', 'r') as f:
        true_classes, predicted_classes, probabilities, class_names = json.load(f)
    save_performance_metrics(true_classes, predicted_classes, probabilities, class_names, tmpdir)
    assert Path(tmpdir, 'class_metrics.html').exists()
    assert Path(tmpdir, 'general_metrics.html').exists()
    assert Path(tmpdir, 'conf_matrix.html').exists()
    assert Path(tmpdir, 'roc.pdf').exists()