import numpy as np

def trajectoryDesign(param):
    # Generate waypoint intercept times and data
    t, X = generateWaypoints(param)

    # Generate trajectory structure from waypoint data
    traj = generateTrajectoryFromWaypoints(t, X)

    return traj

def generateWaypoints(param):
    nWaypoints = 16
    t = np.zeros(nWaypoints)
    X = np.zeros((2, 2, nWaypoints))

    # Waypoint 0
    t[0] = 0
    X[:, :, 0] = [
        [1.1, 0],
        [15, 0]
    ]

    # Waypoint 1
    t[1] = 2.5
    X[:, :, 1] = [
        [6, 0.5],
        [15, 0]
    ]

    # Waypoint 2
    t[2] = 4
    X[:, :, 2] = [
        [7, 0],
        [14, -0.5]
    ]

    # Waypoint 3
    t[3] = 5
    X[:, :, 3] = [
        [7, 0],
        [12.5, -0.5]
    ]

    # Waypoint 4
    t[4] = 6.5
    X[:, :, 4] = [
        [6, -1.5],
        [11.5, 0]
    ]

    # Waypoint 5
    t[5] = 8.5
    X[:, :, 5] = [
        [2, -0.5],
        [11.5, 0]
    ]

    # Waypoint 6
    t[6] = 10
    X[:, :, 6] = [
        [0.75, 0],
        [12.75, 0.5]
    ]

    # Waypoint 7
    t[7] = 12
    X[:, :, 7] = [
        [1.5, 0.5],
        [13.5, 0]
    ]

    # Waypoint 8
    t[8] = 14
    X[:, :, 8] = [
        [1.8, 0],
        [12, -0.5]
    ]

    # Waypoint 9
    t[9] = 16
    X[:, :, 9] = [
        [3.5, 0.5],
        [11.5, 0]
    ]

    # Waypoint 10
    t[10] = 18
    X[:, :, 10] = [
        [6, 0.5],
        [11.5, 0]
    ]

    # Waypoint 11
    t[11] = 20
    X[:, :, 11] = [
        [7, 0],
        [10, -0.5]
    ]

    # Waypoint 12
    t[12] = 23
    X[:, :, 12] = [
        [7, 0],
        [4.5, -0.5]
    ]

    # Waypoint 13
    t[13] = 25
    X[:, :, 13] = [
        [6, -0.5],
        [3.3, 0]
    ]

    # Waypoint 14
    t[14] = 27
    X[:, :, 14] = [
        [4, -1],
        [3.3, 0]
    ]

    # Waypoint final
    t[15] = 29.5
    X[:, :, 15] = [
        [0.75, 0],
        [3.3, 0]
    ]

    return t, X

class generateTrajectoryFromWaypoints:
    def __init__(self, t, X):
        # Ensure number of intercept times and waypoint pages match
        assert len(t) == X.shape[2], "Number of times must match number of waypoint pages"

        self.t = np.asarray(t)
        self.X = np.asarray(X)

        self.nDim = X.shape[0]                     # Dimension of splines
        self.nOrder = 2 * X.shape[1] - 1            # Order of polynomial
        self.nWpts = len(t)                         # Number of waypoints
        self.nSplines = self.nWpts - 1              # Number of splines

        # Derivative operator (diag(1:nOrder, -1) in MATLAB)
        self.D = np.diag(np.arange(1, self.nOrder + 1), k=-1)

        # Build T array
        self.T = np.zeros((self.nOrder + 1,
                           (self.nOrder + 1) // 2,
                           self.nWpts))

        for i in range(self.nWpts):
            # First column: t(i)^(0:nOrder)
            self.T[:, 0, i] = self.t[i] ** np.arange(0, self.nOrder + 1)
            # Subsequent columns: apply derivative operator
            for j in range((self.nOrder - 1) // 2):
                self.T[:, j + 1, i] = self.D @ self.T[:, j, i]

        # Coefficient matrix
        self.C = np.zeros((self.nDim, self.nOrder + 1, self.nSplines))
        for i in range(self.nSplines):
            X_block = np.hstack((self.X[:, :, i], self.X[:, :, i + 1]))
            T_block = np.hstack((self.T[:, :, i], self.T[:, :, i + 1]))
            self.C[:, :, i] = X_block @ np.linalg.inv(T_block)