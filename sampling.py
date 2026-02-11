import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sampling(x_limit, y_limit, z_limit, Natom, fixed_plane = ' ' ):
    '''

    Parameters
    ----------
    x_limit : limitation along x-axis ( x boundary )
    y_limit : limitation along y-axis ( y boundary )
    z_limit : limitation along z-axis ( z boundary )
    Natom : number of atoms
    fixed_plane : if 'xy', returned position has all z=0 values. Same rule applies to other condition.
    Notes
    -----
    Position scale will be determined by units of x_limit and eps.
    Ex: if x_limit in m then the scale is mm scale
    Returns
    -------
    pos_dist (N, 3) : position distribution limited by each axis limitation.
    '''
    rng = np.random.default_rng()
    x = rng.uniform(-x_limit, x_limit, size = Natom)
    y = rng.uniform(-y_limit, y_limit, size = Natom)
    z = rng.uniform(-z_limit, z_limit, size = Natom)
    if fixed_plane == ' ':
        pass
    elif fixed_plane == 'xy':
        z = np.zeros(Natom)
    elif fixed_plane == 'yz':
        x = np.zeros(Natom)
    else :
        y = np.zeros(Natom)
    return np.stack([x, y, z], axis = 0).squeeze()


if __name__ == '__main__':
    x_limit = 1
    y_limit = 1
    z_limit = 1
    Natom = 500
    R = sampling(x_limit, y_limit, z_limit, Natom, fixed_plane = 'xy')
    print(R.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for N in np.arange(Natom):
        ax.scatter(R[N,0], R[N,1], R[N,2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
