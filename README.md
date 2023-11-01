# Vocalisations of Disgust
In the Vocalisations of Disgust project, we classify audio and video segments into either 'moral' or 'pathogen' disgust types using computer vision.
We use VideoMAE and XCLIP as feature extractors, and use randomforest and xgboost as a classifier using these features as input.

# Data setup
We expect our data to be in a single `metadata.csv` file and a single folder containing all the video files.
The metadata file should look like:
```csv
VideoID,Disgust category
v_ca_001_01,pathogen disgust
v_ca_001_02,pathogen disgust
v_ca_013_05,moral disgust
v_ca_013_06,moral disgust
v_ca_013_16,pathogen disgust
```

The folder containing the videos should have videos with file names `<VideoID>.mp4' or other kind of video format.

# Installation
In the root of the folder, run:
```shell
pip install .
```

# Inspect video durations (optional)
We can create a plot of the duration of the videos by running:
```
python disgust/video_statistics.py ./Videos
```
A couple of plots will be saved in the videos folder.
Example output:
![_durations_10_20](https://github.com/vocalisations/disgust/assets/6087314/2392c44d-ca0a-4350-9982-ca1a7dd6c973)

# Defacing (optional)
To prevent our model from learning facial expressions and force it to learn on the context of the video, we deface all videos.

```shell
python disgust/deface_all_videos.py ./Videos ./Videos_defaced
```

# Compute features
For every video, features need to be computed. Two feature extractors are supported, VideoMAE and XCLIP. Features need to be calculated only once, as they will be stored in the same folder as the meta data file. As feature extraction can be halted and resumed at your convenience as data is saved after every encoded video. When (re)starting feature extraction, the script will look for existing features and will skip those videos. Features for VideoMAE and XCLIP are processed and stored independently in seperate files. To start extracting features run:
```
python disgust/encode_videos.py metadata.csv ./Videos_defaced xclip
```
or
```
python disgust/encode_videos.py metadata.csv ./Videos_defaced videomae
```

# Train and evaluate classifier
To train and evaluate a classifier on the extracted features run:
```
python disgust/train_classifier.py metadata.csv ./Videos_defaced <feature_extractor_type> <classifier_type>
```
Where `feature_extractor_type` is one of `['xclip', 'videomae', 'all']`, where `all` means using both feature types concatenated as a single feature vector. For `classifier_type` choose one of `['xgboost', 'pcaxgboost', 'pcarf', 'rf']'.
For example:
```
python disgust/train_classifier.py metadata.csv ./Videos_defaced videomae rf
```

Some output will be printed to the command line for quick inspection. The complete output however, is saved in a `performance` folder, created inside the folder that contains the metadata file.
