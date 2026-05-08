import numpy as np
from controlParameters import ControlParameters
from rigidBodyMassMatrix import rigidBodyMassMatrix
from robotConstraints import robotConstraints
from actuatorConfiguration import actuatorConfiguration
from skew import skew

class RobotParameters:
    def __init__(self, controller=None, scenario="Default", dt=0.1, T=5, useGO=False):
        # Modifier flags
        self.controller = controller
        self.scenario = scenario
        self.dt = dt
        self.T = T
        self.useGradientOpt = useGO

        # Control parameters
        self.ctrl = ControlParameters(self.scenario)

        # Geometric parameters
        self.l = 0.2  # Longitudinal distance from B to wheels [m]
        self.d = 0.3  # Distance between wheels [m]
        self.c = 0.1  # Longitudinal distance from B to C [m]
        self.s = 0.05 # Longitudinal distance from B to S [m]

        # Position vectors
        self.rSBb = np.array([-self.s, 0, 0]).reshape(3,1)
        self.rLBb = np.array([self.l, -self.d/2, 0]).reshape(3,1)  # Implement this in Problem 1
        self.rRBb = np.array([self.l, self.d/2, 0]).reshape(3,1)   # Implement this in Problem 1
        self.rCBb = np.array([self.c, 0, 0]).reshape(3,1)          # Implement this in Problem 2

        # Motor parameters
        self.Km = 0.014  # Motor constant [N.m/A] or [V.s/rad]
        self.Ra = 0.657  # Motor armature resistance [Ohm]
        self.Ng = 5      # Motor gear ratio [-]
        self.rw = 0.04   # Wheel radius [m]

        # Physical properties
        self.m = 3       # Robot mass [kg]
        self.Iz = 3 * 0.15**2  # Yaw inertia [kg.m^2]
        self.g = 3.71    # Acceleration due to gravity on Mars [m/s^2]
        self.IBb = np.diag([0, 0, self.Iz])

        # Degrees of freedom
        self.dofIdx = [0, 1, 5]  # N, E, psi (Python uses zero-based indexing)

        # Compute matrices
        self.M3 = self.compute_mass_matrix()
        self.Bc, self.Bcp = self.compute_constraints()
        self.Ba3 = self.compute_actuator_configuration()
        self.Mr = self.Bcp @ self.M3 @ self.Bcp.T

        self.paynuss = np.vstack([self.Bc.T @ np.linalg.inv(self.M3), self.Bcp])
        self.G = np.vstack([self.Bcp @ self.Ba3, np.zeros((3, 2))])

        # Damping matrix
        self.D3 = self.compute_damping_matrix()

    def compute_mass_matrix(self):
        M6 = rigidBodyMassMatrix(self)  # Placeholder function
        return M6[np.ix_(self.dofIdx, self.dofIdx)]

    def compute_constraints(self):
        return robotConstraints(self)  # Placeholder function

    def compute_actuator_configuration(self):
        Ba6 = actuatorConfiguration(self)  # Placeholder function
        return Ba6[np.ix_(self.dofIdx, range(Ba6.shape[1]))]

    def compute_damping_matrix(self):
        SrLBb = skew(self.rLBb)
        SrRBb = skew(self.rRBb)
        D6 = ((self.Km**2 * self.Ng**2) / (self.Ra * self.rw**2)) * np.block([
            [2*np.eye(3), -SrLBb - SrRBb],
            [SrLBb + SrRBb, -np.dot(SrLBb, SrLBb) - np.dot(SrRBb, SrRBb)]
        ])
        return D6[np.ix_(self.dofIdx, self.dofIdx)]
