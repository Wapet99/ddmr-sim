import numpy as np

def robotInitialConditions(param):
    match param.scenario:
        case 'track':
            return np.array([0,0,1,15,0])
        case 'dock':
            return np.array([0,0,0.65,3.3,0])
        case _:
            raise ValueError(f"Invalid scenario: {param.scenario}")