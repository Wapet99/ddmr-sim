from robotDynamics import robotDynamics
def robotDynamicsAdapter(t, x, u, param):
    dx, _, _ = robotDynamics(t, x, u, param, simple=True)
    return dx