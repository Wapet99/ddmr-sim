import numpy as np
class ControlParameters:
    def __init__(self, scenario='track'):
        if scenario not in ['track', 'dock']:
            raise ValueError("Unknown mode")

        self.scenario = scenario

        # Horizons
        if scenario == 'track':
            self.nControlHorizon = 5  # Length of control horizon
            self.nPredictionHorizon = 5
        elif scenario == 'dock':
            self.nControlHorizon = 3
            self.nPredictionHorizon = 50

        self.nSubsteps = 1  # Number of RK substeps per time step

        # Actuator bounds
        self.uLowerBound = np.array([-6, -6]).reshape(2,1)  # Lower actuator bounds at each time step
        self.uUpperBound = np.array([6, 6]).reshape(2,1)  # Upper actuator bounds at each time step

        # Cost penalties
        self.qr = 10000  # Position error penalty
        self.qpsi = 10  # Heading error penalty
        self.ru = 0.1  # Actuator penalty