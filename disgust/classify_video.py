from disgust.models.available_models import available_models


def infer(video_file: str, return_classifications=False, return_logits=True, model=None):
    """Infer classifications/logits of a model for a given video.

    Args:
        model: choose from xclip or videomae
        return_logits:
        return_classifications:
        video_file: path to a video file
    """
    if model not in available_models:
        raise ValueError(f'Invalid method "{model}" selected. Choose from {available_models.keys()}')

    return available_models[model](video_file, return_classifications, return_logits)
