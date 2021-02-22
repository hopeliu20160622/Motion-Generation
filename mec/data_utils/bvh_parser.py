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
        emotion_train, emotion_test = train_test_split(emotion_list, test_size=0.2, shuffle=True)
        train_files.extend(emotion_train)
        test_files.extend(emotion_test)

    return train_files, test_files
