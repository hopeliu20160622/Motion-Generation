from matplotlib.pyplot import scatter
from mg.vis.scatter_animation import scatter_animation
import numpy as np

def test_scatter_animation():
    vis_original = np.load('vis_sample/test.npy')
    scatter_animation(vis_original, 'vis_test_original.mp4')

    # vis_generated = np.load('vis_sample/test_generated.npy')
    # scatter_animation(vis_generated, 'test_generated.gif')
    # pass