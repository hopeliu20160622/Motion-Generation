import pandas as pd
from mec.data_utils.bvh_parser import extract_train_test_path
from PyMO.pymo.parsers import BVHParser


def test_train_test_split():
    train_files, test_files = extract_train_test_path(meta_file='data/emotion-mocap/file-info.csv', filename='filename', target='emotion')
    assert len(train_files) + len(test_files) == pd.read_csv('data/emotion-mocap/file-info.csv').shape[0]
