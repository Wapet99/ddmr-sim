import numpy as np
def robotConstraints(param):
    l = param.l
    Bc = np.array([0,1,l]).reshape(3,1)
    Bcp = np.array([[0,-l,1], [-1,0,0]])
    return Bc, Bcp