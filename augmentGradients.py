import numpy as np

def augmentGradients(f, t, X, U, param):
    dX = np.zeros_like(X)

    x = X[:, 0]
    u = U[:, 0]
    dx, Jx, Ju = f(t, x, u, param)

    dX[:, 0] = dx
    dX[:, 1:] = Jx @ X[:, 1:] + Ju @ U[:, 1:]

    return dX