import os

import numpy as np
import pandas as pd
import pytest
import torch
from mg.data.preprocess import Preprocessor
from mg.data.util import (extract_train_test_path, find_maximum_frame_length, make_seq_list,
                          get_frame_length, joi, joi_to_colnames)
from pymo.parsers import BVHParser
from pytorch3d.transforms import (euler_angles_to_matrix,
                                  matrix_to_euler_angles, matrix_to_quaternion,
                                  matrix_to_rotation_6d)


@pytest.fixture
def ext_train_test_files():
    train_files, test_files = extract_train_test_path(meta_file='data/emotion-mocap/file-info.csv', filename='filename', target='emotion', test_size=0.1)
    return train_files, test_files


def test_train_test_split(ext_train_test_files):
    train_files = ext_train_test_files[0]
    test_files = ext_train_test_files[1]
    assert len(train_files) + len(test_files) == pd.read_csv('data/emotion-mocap/file-info.csv').shape[0]


def test_get_frame_length():
    sample_file = 'data/emotion-mocap/BVH/M09/M09A0V1.bvh' # has 1240 frames
    length = get_frame_length(sample_file)
    assert length == 1240

def test_find_maximum_frame_length(ext_train_test_files):
    test_max = find_maximum_frame_length(ext_train_test_files[1])
    assert test_max == 1922

def test_joi_validity(ext_train_test_files):
    _, test_files = ext_train_test_files[0], ext_train_test_files[1]

    positions, rotations = joi_to_colnames(joi)
    parser = BVHParser()
    parsed = parser.parse(test_files[0])
    seq_df = parsed.values
    for position in positions:
        assert position in seq_df.columns
    for rotation in rotations:
        assert rotation in seq_df.columns

def test_euler_to_matrix():
    test_hips_rot = torch.Tensor([1.755432, -11.232321, -0.278222])
    mat = euler_angles_to_matrix(euler_angles=test_hips_rot, convention='XYZ')
    rot_6d = matrix_to_quaternion(mat)

def test_preprocess_bvh_quaternion(ext_train_test_files):
    _, test_files = ext_train_test_files[0], ext_train_test_files[1]
    preprocessor = Preprocessor(test_files[0], rot_format='quaternion')
    pos_mat, rot_mat = preprocessor.process()
    assert pos_mat.shape[0] == rot_mat.shape[0]
    assert pos_mat.shape[1]/3 == rot_mat.shape[1]/4
    
def test_preprocess_bvh_6d(ext_train_test_files):
    _, test_files = ext_train_test_files[1], ext_train_test_files[1]
    preprocessor = Preprocessor(test_files[1], rot_format='6d')
    pos_mat, rot_mat = preprocessor.process()
    assert pos_mat.shape[0] == rot_mat.shape[0]
    assert pos_mat.shape[1]/3 == rot_mat.shape[1]/6

def test_npz_validity():
    path = os.path.join('npz_data', 'quaternion')
    if not os.listdir(path):
        pass
    npz_files = os.listdir(path)

    for i in npz_files[:20]:
        npz_quat = np.load(os.path.join(path, i))
        pos = npz_quat['pos']
        rot = npz_quat['rot']
        assert pos.shape[0] == rot.shape[0]
        assert pos.shape[1]/3 == rot.shape[1]/4

@pytest.fixture
def test_make_batch():
    files = ['npz_data/quaternion/F01A0V1.npz', 'npz_data/quaternion/M11SU0V1.npz', 'npz_data/quaternion/M11SU4V1.npz',
              'npz_data/quaternion/M11H2V1.npz', 'npz_data/quaternion/M11D3V1.npz']
    mats = make_seq_list(files)
    assert len(mats) == 5
    return mats

def test_sort_batch(test_make_batch):
    mats = test_make_batch
    seq_lengths = [mat.shape[0] for mat in mats]
    sorted_mats = sorted(mats, key=lambda x: x.shape[0], reverse=True)
    assert sorted_mats[0].shape[0] == np.max(seq_lengths) # maximum
    assert sorted_mats[-1].shape[0] == np.min(seq_lengths) # minimum
