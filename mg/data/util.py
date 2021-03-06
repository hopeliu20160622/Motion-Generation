import os
from typing import List

import numpy as np
import pandas as pd
import torch
from pytorch3d.transforms import (euler_angles_to_matrix, matrix_to_quaternion,
                                  matrix_to_rotation_6d)
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d


def convert_angle_representation(col_names: List, motion_values: np.array, angle_representation: str):
    """Convert given motion to desired angle representation.

    # NOW, IT SIMPLY ASSUME MOTION VALUES ARE ORDERED AS ZYX rotation order.
    Args:
        col_names (List): channel names (e.g. ['Hips_Xposition', ..., 'RihtThumb_Yrotation', 'RightThumb_Xrotation'])
        motion_values (np.array): motion values correspond to col_names
        angle_representation (str): target angle representation. Supports ['6d', 'quaternion']

    Returns:
        (col_name, motion_vector) (tuple): converted column name and motion vector array
    """
    # TODO: should support various euler order, using skeleton structure might be better.
    # TODO: Too lengthy. Break this function into several subroutines.

    new_col = []
    converted_angle = []
    for idx in range(0, motion_values.shape[0], 3):
        joint_name = col_names[idx].split('_')[0]
        joint_col = col_names[idx: idx + 3]
        joint_euler = motion_values[idx: idx + 3]

        base = torch.zeros((3,3))
        base[0][0] = joint_euler[2] # Xrotation
        base[1][1] = joint_euler[1] # Yrotation
        base[2][2] = joint_euler[0] # Zrotation
        base = torch.deg2rad(base)  # Convert from degree to radian since PyMO returns with degree

        x_rot = euler_angles_to_matrix(base[0], convention='XYZ')
        y_rot = euler_angles_to_matrix(base[1], convention='XYZ')
        z_rot = euler_angles_to_matrix(base[2], convention='XYZ')

        rot_mat = torch.eye(3)
        channel_order = 'ZYX'
        for axis in channel_order:
            if axis == 'X' :
                rot_mat = torch.matmul(rot_mat, x_rot)
            elif axis == 'Y':
                rot_mat = torch.matmul(rot_mat, y_rot)
            elif axis == 'Z' :
                rot_mat = torch.matmul(rot_mat, z_rot)
            else:
                raise ValueError(f'Wrong channel order given (Capital Only): {channel_order}')

        if angle_representation == 'quaternion':
            new_col_name = [joint_name + '_' + i for i in 'rijk']
            converted = matrix_to_quaternion(rot_mat).numpy()
        elif angle_representation == '6d':
            new_col_name = [joint_name + '_' + i for i in ['6d_0', '6d_1', '6d_2', '6d_3', '6d_4', '6d_5']]
            converted = matrix_to_rotation_6d(rot_mat).numpy()
        else:
            raise ValueError("Undefined angle representation format")
        
        new_col.extend(new_col_name)
        converted_angle.append(converted)
    
    return (new_col, np.concatenate(converted_angle))


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

def make_seq_list(files):
    """Converts from file paths to list of tensors (seq X embedding)"""
    mats = []
    for fn in files:
        mat = np.load(fn)
        input_mat =np.concatenate([mat['pos'], mat['rot']], axis=1)
        input_mat = torch.Tensor(input_mat)
        mats.append(input_mat)
    return mats

def split_seq_X_y(sorted_mat: List):
    x = []
    y = []
    for mat in sorted_mat:
        x.append(mat[:-1,:])
        y.append(mat[1:,:])
    return x, y

def convert_bvh_path_to_npz(files, bvh_dir="npz_data/quaternion"):
    npz_path = []
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        npz_name = file_name + ".npz"
        npz_path.append(os.path.join(bvh_dir, npz_name))
    return npz_path

def make_padded_batch(files):
    """convert from files to padded X, y batch. Notice taht order is sorted by length"""
    seq_list = make_seq_list(files)
    seq_lengths = sorted([seq.shape[0] - 1 for seq in seq_list], reverse=True)
    seq_sorted = sorted(seq_list, key=lambda x: x.shape[0], reverse=True)

    X, y = split_seq_X_y(seq_sorted)
    X_padded = pad_sequence(X, batch_first=True)
    y_padded = pad_sequence(y, batch_first=True)

    return X_padded, y_padded, seq_lengths