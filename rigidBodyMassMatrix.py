import numpy as np
from skew import skew

def rigidBodyMassMatrix(param):
    m = param.m
    IBb = param.IBb
    rCBb = param.rCBb
    SrCBb = skew(rCBb)

    MRB = np.block([
        [m * np.eye(3), -m * SrCBb],
        [m * SrCBb, IBb]
    ])
    return MRB