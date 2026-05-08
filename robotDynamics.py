import numpy as np

def robotDynamics(t, x, u, param, simple=False):
    #Reduced momentum states
    # Ensure shapes
    x = x.flatten()
    # Extract Params
    Mr = param.Mr
    G = param.G
    c = param.c
    m = param.m
    l = param.l
    Iz = param.Iz
    D3 = param.D3
    # Compute intermediate values
    p32 = x[0] * (c * m - l * m) / (m * (l**2) - 2 * c * m * l + Iz)
    dp = (c * m - l * m) / (m * (l**2) - 2 * c * m * l + Iz)
    # Compute dynamics
    fx = np.vstack([
        p32 * x[1] / Mr[1, 1] - x[0] * (D3[0, 0] * (l**2) - 2 * D3[1, 2] * l + D3[2, 2]) / Mr[0, 0],
        -D3[0, 0] * x[1] / Mr[1, 1] - p32 * x[0] / Mr[0, 0],
        x[0] * l * np.sin(x[4]) / Mr[0, 0] - x[1] * np.cos(x[4]) / Mr[1, 1],
        -x[1] * np.sin(x[4]) / Mr[1, 1] - x[0] * l * np.cos(x[4]) / Mr[0, 0],
        x[0] / Mr[0, 0]
    ])
    dx = fx + (G @ u).reshape(-1,1)
    # print(f"fx: {fx}")
    # print(f"G@u: {G@u}")
    # print(f"dx: {dx}")
    Jx = []
    Ju = []
    if not simple:
        Jx = np.array([
            [dp * x[1] / Mr[1, 1] - (D3[0, 0] * (l**2) - 2 * D3[1, 2] * l + D3[2, 2]) / Mr[0, 0], p32 / Mr[1, 1], 0, 0, 0],
            [-2 * p32 / Mr[0, 0], -D3[0, 0] / Mr[1, 1], 0, 0, 0],
            [l * np.sin(x[4]) / Mr[0, 0], -np.cos(x[4]) / Mr[1, 1], 0, 0, x[0] * l * np.cos(x[4]) / Mr[0, 0] + x[1] * np.sin(x[4]) / Mr[1, 1]],
            [-l * np.cos(x[4]) / Mr[0, 0], -np.sin(x[4]) / Mr[1, 1], 0, 0, -x[1] * np.cos(x[4]) / Mr[1, 1] + x[0] * l * np.sin(x[4]) / Mr[0, 0]],
            [1 / Mr[0, 0], 0, 0, 0, 0]
        ])
        Ju = G
    return dx.flatten(), Jx, Ju