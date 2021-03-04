from mg.vis.scatter_animation import scatter_animation
import numpy as np

def test_scatter_animation():
    vis_sample = np.load('vis_sample/F02A4V1_pos.npy')
    scatter_animation(vis_sample, 'F02A4V1_pos.mp4')
    pass