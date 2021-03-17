import os

import numpy as np
import pandas as pd
import pytest
import torch
from mg.data.preprocess import MotionDataset, Preprocessor
from mg.data.util import (convert_bvh_path_to_npz, extract_train_test_path,
                          find_maximum_frame_length, get_frame_length, joi,
                          joi_to_colnames, make_padded_batch, make_seq_list,
                          split_seq_X_y, convert_angle_representation)
from pymo.parsers import BVHParser
from pytorch3d.transforms import (euler_angles_to_matrix,
                                  matrix_to_euler_angles, matrix_to_quaternion,
                                  matrix_to_rotation_6d)
from torch.utils.data import DataLoader
from mg.data.preprocess import BVHProcessor

@pytest.fixture
def ext_train_test_files():
    train_files, test_files = extract_train_test_path(meta_file='data/emotion-mocap/file-info.csv', filename='filename', target='emotion', test_size=0.1)
    return train_files, test_files


def test_convert_angle_representation():
    cmu_path = 'data/cmu-mocap/data/001/01_01.bvh'
    bvh_processor = BVHProcessor(cmu_path)
    motion_joint_num = (bvh_processor.motion.shape[1] - 3) / 3

    new_col_names_0, converted_vector_0 = bvh_processor.make_state_input_angle_by_frame_id(frame_id=0, angle_representation='quaternion')
    
    assert len(new_col_names_0) / 4 == motion_joint_num
    assert converted_vector_0.shape[0] / 4 == motion_joint_num

    new_col_names_1, converted_vector_1 = bvh_processor.make_state_input_angle_by_frame_id(frame_id=1, angle_representation='quaternion')
    assert len(new_col_names_1) / 4 == motion_joint_num
    assert converted_vector_1.shape[0] / 4 == motion_joint_num

    new_col_names_6d, converted_vector_6d = bvh_processor.make_state_input_angle_by_frame_id(frame_id=20, angle_representation='6d')
    assert len(new_col_names_6d) / 6 == motion_joint_num
    assert converted_vector_6d.shape[0] / 6 == motion_joint_num

def test_root_velocity():
    cmu_path = 'data/cmu-mocap/data/001/01_01.bvh'
    bvh_processor = BVHProcessor(cmu_path)
    
    root_v = bvh_processor.make_state_input_root_v_by_frame_id(0)
    assert root_v.shape[0] == 3
    np.testing.assert_array_equal(root_v, np.array([0,0,0]))

    root_v_2 = bvh_processor.make_state_input_root_v_by_frame_id(2)
    true_root_v_2 = bvh_processor.motion.iloc[2].values[:3] - bvh_processor.motion.iloc[1].values[:3] 
    np.testing.assert_array_equal(root_v_2, true_root_v_2)

    root_v_40 = bvh_processor.make_state_input_root_v_by_frame_id(40)
    true_root_v_40 = bvh_processor.motion.iloc[40].values[:3] - bvh_processor.motion.iloc[39].values[:3] 
    np.testing.assert_array_equal(root_v_40, true_root_v_40)
    

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
    preprocessor = Preprocessor(test_files[1], rot_format='quaternion')
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

def test_split_seq_train_test(test_make_batch):
    mats = test_make_batch
    seq_lengths = [mat.shape[0] for mat in mats]
    sorted_mats = sorted(mats, key=lambda x: x.shape[0], reverse=True)
    assert sorted_mats[0].shape[0] == np.max(seq_lengths) # maximum
    assert sorted_mats[-1].shape[0] == np.min(seq_lengths) # minimum
    
    x, y = split_seq_X_y(sorted_mats)
    assert sorted(seq_lengths, reverse=True) == [916, 854, 788, 758, 748]
    assert x[0].size() == (915, 147)
    assert torch.all(x[0][914] == y[0][913]).item() == True
    assert x[1].size() == (853, 147)
    assert torch.all(x[1][852] == y[1][851]).item() == True
    assert x[-1].size() == (747, 147)
    assert torch.all(x[-1][746] == y[-1][745]).item() == True
    assert torch.all(x[-1][241] == y[-1][240]).item() == True

@pytest.fixture
def test_make_padded_batch(ext_train_test_files):
    _, test_files = ext_train_test_files[0], ext_train_test_files[1]
    test_files_npz = convert_bvh_path_to_npz(test_files)
    X_padded, y_padded, seq_lengths = make_padded_batch(test_files_npz)
    assert X_padded.shape[0] == y_padded.shape[0]
    assert X_padded.shape[1] == y_padded.shape[1] == seq_lengths[0]
    return X_padded, y_padded, seq_lengths

def test_data_loader(test_make_padded_batch):
    X_padded, y_padded, seq_lengths = test_make_padded_batch
    dataset = MotionDataset(X_padded, y_padded, seq_lengths)
    loader = DataLoader(dataset, batch_size=5)
    data_count = 0
    for batch in loader:
        data_count += batch[0].shape[0]
    assert data_count == X_padded.shape[0]
