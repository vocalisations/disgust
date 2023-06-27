from itertools import chain
from typing import List

from disgust.models.available_models import available_models


def infer(video_file: str, return_classifications=False, return_logits=True, model=None):
    """Infer classifications/logits of a model for a given video.

    Args:
        model: choose from xclip or videomae
        return_logits:
        return_classifications:
        video_file: path to a video file
    """
    return get_model(model).infer(video_file, return_classifications, return_logits)


def get_feature_names(model) -> List[str]:
    """Get a list of the names of the features that the model outputs."""
    models = available_models.keys() if model == "all" else [model]
    return chain.from_iterable([get_model(model).get_feature_names() for model in models])


def get_model(model):
    if model not in available_models:
        raise ValueError(f'Invalid method "{model}" selected. Choose from {available_models.keys()}')
    return available_models[model]
