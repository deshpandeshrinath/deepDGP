import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def ax3d(title='', axis=''):
    fig, ax = plt.subplots()
    ax = fig.gca(projection='3d')
    ax.set_title(title)
    if axis=='equal':
        ax.axis('equal')
    return fig, ax

def ax2d(title='', axis=''):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(title)
    if axis=='equal':
        ax.axis('equal')
    return fig, ax

def plot_3d_poses(poses, ax):
    """ poses must be in shape [batch, num, 3]
    """
    for pose in poses:
        ax.plot(pose[:,0], pose[:,1], pose[:,2])

def plot_2d_poses(poses, ax):
    """ poses must be in shape [batch, num, 3]
    """
    for pose in poses:
        ax.plot(pose[:,0], pose[:,1])


