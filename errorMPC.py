import numpy as np
from robotDynamics import robotDynamics
from augmentGradients import augmentGradients
from trajectoryEval import trajectoryEval

def errorMPC(t1, x1, u0, U, param):
    dt = param.dt / param.ctrl.nSubsteps
    Np = param.ctrl.nPredictionHorizon
    Nc = param.ctrl.nControlHorizon
    assert Np >= Nc, "Prediction horizon cannot be less than control horizon"

    qr = param.ctrl.qr
    qpsi = param.ctrl.qpsi
    ru = param.ctrl.ru

    nx = len(x1)
    nu = len(u0)
    e = np.zeros(5 * Np)

    t = t1
    x = x1
    f = robotDynamics

    J = None
    if U is None:
        U = np.zeros(nu * Nc)

    if param.useGradientOpt:
        J = np.zeros((5 * Np, nu * Nc))
        dxdU = np.zeros((nx, nu * Nc))
        XX = np.hstack([x.reshape(-1, 1), dxdU])
        #print(x)
        #print(XX)

    for k in range(1, Np + 1): #Each step in prediction horizon
        u = U[(nu * (k - 1)):nu * k] if k <= Nc else np.array([0, 0])

        for j in range(param.ctrl.nSubsteps): #nSubsteps of Runge-Kutta integration using 3/8 rule
            if J is None:
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

        X = np.squeeze(trajectoryEval(param.traj, t))
        # print("X type:", type(X)) 
        # print("X shape:", np.shape(X)) 
        # print("X[0,0] =", X[0,0])
        rPNn = np.array([X[0, 0], X[1, 0], 0])
        vPNn = np.array([X[0, 1], X[1, 1], 0])
        
        psistar = x[4] if np.linalg.norm(vPNn) == 0 else np.arctan2(vPNn[1], vPNn[0])
        psistar += 2 * np.pi * round((x[4] - psistar) / (2 * np.pi))
        
        rCNn = np.array([param.c * np.cos(x[4]) + x[2], param.c * np.sin(x[4]) + x[3], 0])
        pos = rCNn - rPNn
        segment = np.concatenate([
            np.sqrt(qr) * pos[:2],
            np.array([np.sqrt(qpsi) * (x[4] - psistar)]),
            np.sqrt(ru) * u
        ])
        e[(k - 1) * 5:k * 5] = segment

        if J is not None:
            der = np.sqrt(qr) * np.dot([[0, 0, 1, 0, -param.c * np.sin(x[4])], 
                                        [0, 0, 0, 1, param.c * np.cos(x[4])]], dxdU)
            depsi = np.dot([0, 0, 0, 0, np.sqrt(qpsi)], dxdU)
            if np.linalg.norm(vPNn) == 0:
                depsi = np.zeros_like(depsi)

            deu = np.sqrt(ru) * dudU
            J[(k - 1) * 5:k * 5, :] = np.vstack([der, depsi, deu])

    return (e, J) if J is not None else e