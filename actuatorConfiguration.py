import numpy as np
from skew import skew

def actuatorConfiguration(param):
    Km = param.Km
    Ra = param.Ra
    Ng = param.Ng
    rw = param.rw

    rLBb = param.rLBb
    rRBb = param.rRBb
    e1 = np.array([1,0,0]).reshape(3,1) #or np.array([[1],[2],[3]])

    Ba = (((Ng * Km) / Ra) / rw) * np.block([
        [e1, e1],
        [skew(rLBb) @ e1, skew(rRBb) @ e1]
    ])
    return Ba