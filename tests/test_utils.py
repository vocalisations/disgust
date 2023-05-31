import pytest

from disgust.utils import read_feature_set
import numpy as np


@pytest.mark.parametrize('feature_string, expected', [
    ('[1.2, 1.0]', np.array([1.2, 1.0])),
    ('NaN', None),
    ('nan', None),
    (np.nan, None),
])
def test_read_feature_set(feature_string, expected):
    result = read_feature_set(feature_string)

    if expected is None:
        assert result is None
    else:
        np.testing.assert_allclose(result, expected, atol=1e-5)
