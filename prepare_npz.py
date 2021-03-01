import pathlib
from mg.data.util import extract_train_test_path
from mg.data.preprocess import Preprocessor
import os
import numpy as np

p = pathlib.Path('npz_data/quaternion')
p.mkdir(parents=True, exist_ok=True)
p = pathlib.Path('npz_data/6d')
p.mkdir(parents=True, exist_ok=True)

train_files, test_files = extract_train_test_path(meta_file='data/emotion-mocap/file-info.csv', filename='filename', target='emotion', test_size=0.1)
files = train_files + test_files
for cnt, file in enumerate(train_files + test_files):
    rot_format = 'quaternion'
    preprocessor = Preprocessor(file, rot_format=rot_format)
    pos_mat, rot_mat = preprocessor.process()
    file_name = os.path.splitext(os.path.split(file)[1])[0]
    np.savez(os.path.join('npz_data', rot_format, file_name), pos=pos_mat, rot=rot_mat)
    print(f"Processed: {file_name}. ({cnt+1}/{len(files)})")
