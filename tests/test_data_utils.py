import pandas as pd
from mec.data.util import extract_train_test_path, get_frame_length, find_maximum_frame_length
from mec.data.preprocess import Preprocessor
import pytest

@pytest.fixture
def ext_train_test_files():
    train_files, test_files = extract_train_test_path(meta_file='data/emotion-mocap/file-info.csv', filename='filename', target='emotion')
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

def test_preprocessor(ext_train_test_files):
    train_files, test_files = ext_train_test_files[0], ext_train_test_files[1]
    preprocessor = Preprocessor(test_files, 1922)
    preprocessor.process()
    print("Done")