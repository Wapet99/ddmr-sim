import numpy as np
def skew(vector):
    """ Compute the skew-symmetric matrix for a given 3D vector. """
    vector = vector.flatten()
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])