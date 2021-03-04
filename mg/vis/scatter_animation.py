from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def animate_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.

    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)

    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][0:1], data[iteration][1:2], data[iteration][2:])
    return scatters

def scatter_animation(arr, file_name):

    fig = plt.figure()
    ax = Axes3D(fig)

    # create the parametric curve
    x=arr[:,0]
    y=arr[:,1]
    z=arr[:,2]

    # create the first plot
    point, = ax.plot([x[0]], [y[0]], [z[0]], 'o')
    ax.legend()
    # ax.set_xlim([-1.5, 1.5])
    # ax.set_ylim([-1.5, 1.5])
    # ax.set_zlim([-1.5, 1.5])

    # second option - move the point position at every frame
    def update_point(n, x, y, z, point):
        point.set_data(np.array([x[n], y[n]]))
        point.set_3d_properties(z[n], 'z')
        return point

    ani=animation.FuncAnimation(fig, update_point, 50, fargs=(x, y, z, point))

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, bitrate=900, extra_args=['-vcodec', 'libx264'])
    ani.save(file_name, writer=writer)