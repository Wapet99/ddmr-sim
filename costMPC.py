import numpy as np
from robotDynamics import robotDynamics
from augmentGradients import augmentGradients

def costMPC(t1, x1, u0, U, param):
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

    g = None
    if hasattr(param, 'useGradientOpt') and param.useGradientOpt:
        J = np.zeros((5 * Np, nu * Nc))
        dxdU = np.zeros((nx, nu * Nc))
        XX = np.hstack([x.reshape(-1, 1), dxdU])

    for k in range(1, Np + 1):
        u = U[(nu * (k - 1)):nu * k] if k <= Nc else np.array([0, 0])

        for j in range(param.ctrl.nSubsteps):
            if g is None:
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

        rENn = np.array([0.75, 1.7, 0])
        psistar = 0
        rCNn = np.array([param.c * np.cos(q[2]) + q[0], param.c * np.sin(q[2]) + q[1], 0])
        rCEn = rCNn - rENn

        if t >= 5:
            er = np.sqrt(qr) * rCEn[:2]
            epsi = np.sqrt(qpsi) * (x[4] - psistar)
            eu = np.sqrt(ru) * u
        else:
            er = np.zeros(2)
            epsi = 0
            eu = np.sqrt(ru) * u

        e[(k - 1) * 5:k * 5] = np.concatenate([er, [epsi], eu])

        if g is not None:
            if t >= 5:
                der = np.sqrt(100 * qr) * np.dot([[0, 0, 1, 0, -param.c * np.sin(x[4])], 
                                                  [0, 0, 0, 1, param.c * np.cos(x[4])]], dxdU)
                depsi = np.dot([0, 0, 0, 0, np.sqrt(100 * qpsi)], dxdU)
                deu = np.sqrt(100 * ru) * dudU
                J[(k - 1) * 5:k * 5, :] = np.vstack([der, depsi, deu])
            else:
                der = np.zeros((2, 5)) @ dxdU
                depsi = np.zeros((1, 5)) @ dxdU
                deu = np.sqrt(ru) * dudU
                J[(k - 1) * 5:k * 5, :] = np.vstack([der, depsi, deu])

    V = np.dot(e.T, e)
    g = 2 * J.T @ e if g is not None else None

    return (V, g) if g is not None else V