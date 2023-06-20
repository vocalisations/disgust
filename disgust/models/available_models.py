from disgust.models import classify_video_with_xclip, classify_video_with_videomae

available_models = {
    'xclip': classify_video_with_xclip,
    'videomae': classify_video_with_videomae,
}