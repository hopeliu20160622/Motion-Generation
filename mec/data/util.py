import os

import pandas as pd
from sklearn.model_selection import train_test_split
from PyMO.pymo.parsers import BVHParser

def extract_train_test_path(meta_file: str, filename: str, target: str):
    """meta file should have at least two columns, describing "file name" and "target(emotion)"""
    meta_df = pd.read_csv(meta_file)
    emotions = meta_df[target].unique()

    PREFIX = 'data/emotion-mocap/BVH'
    SUFFIX = '.bvh'

    train_files = []
    test_files = []

    for emotion in emotions:
        emotion_set = meta_df[meta_df[target] == emotion]
        emotion_list = []
        for _, emotion_item in emotion_set.iterrows():
            actor = emotion_item['actor_ID']
            fn = emotion_item[filename]
            path = os.path.join(PREFIX, actor, fn + SUFFIX)
            emotion_list.append(path)
        emotion_train, emotion_test = train_test_split(emotion_list, test_size=0.2, shuffle=True, random_state=42)
        train_files.extend(emotion_train)
        test_files.extend(emotion_test)

    return train_files, test_files


def get_frame_length(file):
    with open(file, "r") as fp:
        searchlines = fp.readlines()
    for i, line in enumerate(searchlines):
        if "Frames:" in line:
            length = int(line.rstrip('\n').split(':')[1])
            return length

def find_maximum_frame_length(files):
    max_len = 0
    for file in files:
        file_len = get_frame_length(file)
        if file_len > max_len:
            max_len = file_len
    return max_len