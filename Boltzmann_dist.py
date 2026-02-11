import numpy as np
import scipy.constants as cts
import matplotlib.pyplot as plt

def vel_Boltzmann(T, m, N = 1, fixed_plane = ' '):
    '''
    Parameters
    ----------
    m : atom's mass (in kg)

    T : temperature (in K)

    fixed_plane : if 'xy', returned position has all z=0 values. Same rule applies to other condition.

    Returns
    -------
    vel : velocity (in m/s)
    '''
    sigma = np.sqrt(cts.k * T / m)
    n = np.random.normal(0, 1.0, size=(3, N))
    if fixed_plane == ' ':
        pass
    elif fixed_plane == 'xy':
        n[2, :] = 0
    elif fixed_plane == 'yz':
        n[0, :] = 0
    else :
        n[1, :] = 0

    v = sigma * n
    return v.squeeze()

if __name__ == '__main__':
    Ts = [100, 300, 1000]
    for T in Ts:
        Vs = vel_Boltzmann(T, 1, 1000, fixed_plane = 'xy')
        magVs = np.array(np.sqrt(Vs[0,:]**2+Vs[1,:]**2+Vs[2,:]**2))
        counts, bin_edges = np.histogram(magVs, bins=100)
        # Plot the histogram using Matplotlib's bar function
        plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', alpha = 0.5, edgecolor='black', label = f'temperature = {T}')
    plt.legend()
    plt.title('NumPy Histogram with Matplotlib')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
