import numpy as np
from augmentGradients import augmentGradients
from robotDynamics import robotDynamics

def nonlconMPC(t1, x1, u0, U, param):
    dt = param.dt
    Np = param.ctrl.nPredictionHorizon
    Nc = param.ctrl.nControlHorizon
    assert Np >= Nc, "Prediction horizon cannot be less than control horizon"

    nx = len(x1)
    nu = len(u0)

    # Nonlinear equality constraints (empty in this case)
    ceq = np.array([])

    # Nonlinear inequality constraints
    cineq = np.zeros(6 * Np)

    # Walls
    NL, NU = 0.4, 3.8
    EL, EU = 1.1, 3.8

    # Obstacles
    rO1Nn = np.array([0.4, 2.5, 0])  # Center of obstacle 1
    r1 = 0.5  # Radius of obstacle 1
    rO2Nn = np.array([1.75, 2.5, 0])  # Center of obstacle 2
    r2 = 0.25  # Radius of obstacle 2

    # Distance between C and furthest extent of robot chassis
    L = 0.25

    t = t1
    x = x1
    f = robotDynamics

    gcineq = None
    gceq = None
    if hasattr(param, 'useGradients') and param.useGradients:
        gcineq = np.zeros((6 * Np, nu * Nc))
        gceq = np.zeros((0, nu * Nc))
        dxdU = np.zeros((nx, nu * Nc))
        XX = np.hstack([x.reshape(-1, 1), dxdU])

    for k in range(1, Np + 1):
        u = U[(nu * (k - 1)):nu * k] if k <= Nc else np.array([0, 0])

        for j in range(param.ctrl.nSubsteps):
            if gcineq is None:
                f1 = f(t, x, u, param)
                f2 = f(t + dt / 3, x + f1 * dt / 3, u, param)
                f3 = f(t + 2 * dt / 3, x - f1 * dt / 3 + f2 * dt, u, param)
                f4 = f(t + dt, x + f1 * dt - f2 * dt + f3 * dt, u, param)
                x += (f1 + 3 * f2 + 3 * f3 + f4) * dt / 8
            else:
                dudU = np.zeros((nu, nu * Nc))
                if k <= Nc:
                    dudU[:, nu * (k - 1): nu * k] = np.eye(nu)

                UU = np.hstack([u.reshape(-1, 1), dudU])
                F1 = augmentGradients(f, t, XX, UU, param)
                F2 = augmentGradients(f, t + dt / 3, XX + F1 * dt / 3, UU, param)
                F3 = augmentGradients(f, t + 2 * dt / 3, XX - F1 * dt / 3 + F2 * dt, UU, param)
                F4 = augmentGradients(f, t + dt, XX + F1 * dt - F2 * dt + F3 * dt, UU, param)
                XX += (F1 + 3 * F2 + 3 * F3 + F4) * dt / 8
                x = XX[:, 0]
                dxdU = XX[:, 1:]

            t += dt

        q = x[2:5]
        rCNn = np.array([param.c * np.cos(q[2]) + q[0], param.c * np.sin(q[2]) + q[1], 0])
        TEMP1 = rCNn - rO1Nn
        TEMP2 = rCNn - rO2Nn

        c1 = NL + L - rCNn[0]
        c2 = -NU + L + rCNn[0]
        c3 = EL + L - rCNn[1]
        c4 = -EU + L + rCNn[1]
        c5 = (r1 + L) - np.linalg.norm(TEMP1)
        c6 = (r2 + L) - np.linalg.norm(TEMP2)

        cineq[(k - 1) * 6:k * 6] = [c1, c2, c3, c4, c5, c6]

        if gcineq is not None:
            dc1dx = np.array([0, 0, -1, 0, param.c * np.sin(q[2])])
            dc2dx = np.array([0, 0, 1, 0, -param.c * np.sin(q[2])])
            dc3dx = np.array([0, 0, 0, -1, -param.c * np.cos(q[2])])
            dc4dx = np.array([0, 0, 0, 1, param.c * np.cos(q[2])])
            dc5dx = np.array([0, 0, -TEMP1[0] / np.linalg.norm(TEMP1), -TEMP1[1] / np.linalg.norm(TEMP1),
                              -param.c * ((rO1Nn[0] - q[0]) * np.sin(q[2]) + (q[1] - rO1Nn[1]) * np.cos(q[2])) / np.linalg.norm(TEMP1)])
            dc6dx = np.array([0, 0, -TEMP2[0] / np.linalg.norm(TEMP2), -TEMP2[1] / np.linalg.norm(TEMP2),
                              -param.c * ((rO2Nn[0] - q[0]) * np.sin(q[2]) + (q[1] - rO2Nn[1]) * np.cos(q[2])) / np.linalg.norm(TEMP2)])
            gcineq[(k - 1) * 6:k * 6, :] = np.vstack([dc1dx, dc2dx, dc3dx, dc4dx, dc5dx, dc6dx]) @ dxdU

    if gcineq is not None:
        gcineq = gcineq.T
        gceq = gceq.T

    return (cineq, ceq, gcineq, gceq) if gcineq is not None else (cineq, ceq)