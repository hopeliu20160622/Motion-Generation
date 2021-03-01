from typing import List


from pymo.parsers import BVHParser
from mg.data.util import joi, joi_to_colnames, euler_to_quaternion, euler_to_6d
import numpy as np

# Need to be processed by length X 1 X embeddings
# Which corresponds to total_frames X 1 X embeddings 

# rotation:
#   * refer: https://www.mathworks.com/help/robotics/ref/eul2rotm.html
#   * refer: https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.euler_angles_to_matrix

class Preprocessor:
    """This processes single bvh file and return position and rotation matrix in numpy format"""
    # reference for parsed bvh: https://wiki.reallusion.com/IC_Python_API:RLPy_RIBodyDevice
    def __init__(self, file: List, rot_format: str=None) -> None:
        self.file = file
        self.parser = BVHParser()
        self.rot_format = rot_format # one of a ['None', 'quaternion', '6d']

    def process(self):
        bvh_parsed = self.parser.parse(self.file)
        bvh_df = bvh_parsed.values
        pos_cols, rot_cols = joi_to_colnames(joi)
        rot_grouped = [rot_cols[i:i+3] for i in range(0, len(rot_cols), 3)] # group XYZ by joint
        rot_convert = []
        for rot_info in rot_grouped:
            if self.rot_format == 'quaternion':
                values = euler_to_quaternion(bvh_df[rot_info].values)
            elif self.rot_format == '6d':
                values = euler_to_6d(bvh_df[rot_info].values)
            else:
                values = bvh_df[rot_cols].values
            rot_convert.append(values)
        rot_matrix = np.concatenate(rot_convert, axis=1)
        return bvh_df[pos_cols].values, rot_matrix

