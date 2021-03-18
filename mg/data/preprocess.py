from typing import List


from pymo.parsers import BVHParser
from mg.data.util import joi, joi_to_colnames, euler_to_quaternion, euler_to_6d, convert_angle_representation
import numpy as np
from torch.utils.data import Dataset


# Need to be processed by length X 1 X embeddings
# Which corresponds to total_frames X 1 X embeddings

# rotation:
#   * refer: https://www.mathworks.com/help/robotics/ref/eul2rotm.html
#   * refer: https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.euler_angles_to_matrix

class BVHProcessor:
    """Parse BVH file and provide useful transform needed for motion generation"""
    def __init__(self, file_name: str):    
        self.file_name = file_name
        self._parsed = BVHParser().parse(file_name)
        self.motion = self._parsed.values

    def make_state_input_angle_by_frame_id(self, frame_id: int, angle_representation: str):
        """Query motion by given frame id and prepare input vector
        Args:
            frame_id (int): frame_id
            angle_representation (str): angle representation in vector form. Supports ['6d', 'quarternion']
        """
        rot_cols = [rot_col for rot_col in self.motion.columns.tolist() if 'rotation' in rot_col] # use rotation values only
        motion_sr = self.motion.iloc[frame_id][rot_cols]
        motion_values = motion_sr.values
        new_col_names, converted_vector = convert_angle_representation(rot_cols, motion_values, angle_representation)
        return new_col_names, converted_vector

    def make_state_input_root_v_by_frame_id(self, frame_id: int):
        motion_sr = self.motion.iloc[frame_id]
        motion_pos = motion_sr.values[:3]

        if frame_id == 0:
            prev_pos = motion_pos
        else:
            prev_pos = self.motion.iloc[frame_id - 1].values[:3]
        
        root_vel = motion_pos - prev_pos
        return root_vel


    def make_offset_input_by_frame_id(self, frame_id: int, target_frame_id: int, angle_representation: str):
        motion_target = self.motion.iloc[target_frame_id]
        target_root_pos = motion_target.values[:3] # target xyz
        target_angle = self.make_state_input_angle_by_frame_id(target_frame_id, angle_representation)

        cur_motion = self.motion.iloc[frame_id]
        cur_motion_pos = cur_motion.values[:3] # current xyz
        cur_angle = self.make_state_input_angle_by_frame_id(frame_id, angle_representation)

        offset_pos = target_root_pos - cur_motion_pos
        offset_angle = target_angle[1] - cur_angle[1]
        
        return (offset_pos, offset_angle)

    def make_target_input_by_frame_id(self, frame_id: int):

        pass


class Preprocessor:
    """This processes single bvh file and return position and rotation matrix in numpy format"""
    # reference for parsed bvh: https://wiki.reallusion.com/IC_Python_API:RLPy_RIBodyDevice

    def __init__(self, file: List, rot_format: str = None) -> None:
        self.file = file
        self.parser = BVHParser()
        self.rot_format = rot_format  # one of a ['None', 'quaternion', '6d']

    def process(self):
        bvh_parsed = self.parser.parse(self.file)
        bvh_df = bvh_parsed.values
        pos_cols, rot_cols = joi_to_colnames(joi)
        rot_grouped = [rot_cols[i:i+3]
                       for i in range(0, len(rot_cols), 3)]  # group XYZ by joint
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


class MotionDataset(Dataset):

    def __init__(self, X_padded, y_padded, seq_lengths):
        self.X_padded = X_padded
        self.y_padded = y_padded
        self.seq_lengths = seq_lengths

    def __len__(self):
        assert self.X_padded.shape[0] == self.y_padded.shape[0]
        return self.X_padded.shape[0]

    def __getitem__(self, index):
        return self.X_padded[index], self.y_padded[index], self.seq_lengths[index]
    