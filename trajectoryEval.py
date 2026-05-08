import numpy as np

def trajectoryEval(traj, t):
    t = np.atleast_1d(t)
    X = np.full((traj.nDim, (traj.nOrder + 1) // 2, len(t)), np.nan)

    for i, ti in enumerate(t):
        if ti < traj.t[0]:
            X[:, :, i] = traj.X[:, :, 0]
            X[:, 1:, i] = 0
            continue
        if ti > traj.t[-1]:
            X[:, :, i] = traj.X[:, :, -1]
            X[:, 1:, i] = 0
            continue
        if ti == traj.t[-1]:
            X[:, :, i] = traj.X[:, :, -1]
            continue

        T = np.zeros((traj.nOrder + 1, (traj.nOrder + 1) // 2))
        T[:, 0] = ti ** np.arange(traj.nOrder + 1)
        for j in range((traj.nOrder - 1) // 2):
            T[:, j + 1] = traj.D @ T[:, j]

        k = np.where(traj.t <= ti)[0][-1]  # Find last index where traj.t <= ti
        X[:, :, i] = traj.C[:, :, k] @ T

    return X