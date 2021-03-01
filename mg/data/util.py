import os

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d
import torch

joi = [
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand"
]

def joi_to_colnames(joi: List):
    """Simply append suffix to make column names compatible with PyMO"""
    positions = []
    rotations = []
    for joint in joi:
        position = [joint + i for i in ["_Xposition", "_Yposition", "_Zposition"]]   
        rotation = [joint + i for i in ["_Xrotation", "_Yrotation", "_Zrotation"]]
        positions.extend(position)
        rotations.extend(rotation)
    return positions, rotations

def euler_to_quaternion(euler):
    """Convert euler(N, 3) representation to quaternion(N, 4) representation"""
    # https://forums.autodesk.com/t5/fbx-forum/eulerangles-quaternions-and-the-right-rotationorder/m-p/4166206/highlight/true#M3108
    euler_vals = torch.Tensor(euler)
    matrix = euler_angles_to_matrix(euler_vals, convention='XYZ')
    quaternion_vals = matrix_to_quaternion(matrix)
    return quaternion_vals.numpy()

def euler_to_6d(euler):
    """Convert euler(N, 3) representation to 6d(N, 6) representation"""
    # https://forums.autodesk.com/t5/fbx-forum/eulerangles-quaternions-and-the-right-rotationorder/m-p/4166206/highlight/true#M3108
    euler_vals = torch.Tensor(euler)
    matrix = euler_angles_to_matrix(euler_vals, convention='XYZ')
    rot6d_vals = matrix_to_rotation_6d(matrix)
    return rot6d_vals.numpy()

def extract_train_test_path(meta_file: str, filename: str, target: str, test_size: float = 0.2):
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
        emotion_train, emotion_test = train_test_split(
            emotion_list, test_size=test_size, shuffle=True, random_state=42)
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
