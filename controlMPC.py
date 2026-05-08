import numpy as np
from scipy.optimize import least_squares, minimize
from errorMPC import errorMPC
from costMPC import costMPC
from nonlconMPC import nonlconMPC

class MPCResidualWithJacobian:
    # wrapper class to mimic lsqnonlin behaviour with scipy
    def __init__(self, t1, x1, u0, param):
        self.t1 = t1
        self.x1 = x1
        self.u0 = u0
        self.param = param
        self.last_U = None
        self.last_e = None
        self.last_J = None

    def evaluate(self, U):
        # Only recompute if U changed
        if self.last_U is None or not np.array_equal(U, self.last_U):
            e, J = errorMPC(self.t1, self.x1, self.u0, U, self.param)
            self.last_U = U.copy()
            self.last_e = e
            self.last_J = J
        return self.last_e, self.last_J

    def fun(self, U):
        e, _ = self.evaluate(U)
        return e

    def jac(self, U):
        _, J = self.evaluate(U)
        return J


def controlMPC(t1, x1, u0, U, param):
    m = u0.shape[0]

    if U is None or len(U) == 0:
        U = np.concatenate([u0.flatten(), np.zeros(m * (param.ctrl.nControlHorizon - 1))])
    else:
        # warm start
        U = np.concatenate([U[m:], np.zeros(m)])

    scenario = param.scenario

    if scenario == 'track':
        options = {
            'verbose': 2,
            'max_nfev': int(1e5),
            'x_scale': 'jac' if param.useGradientOpt else None
        }
        #err = lambda U: errorMPC(t1, x1, u0, U, param)
        lb = np.tile(param.ctrl.uLowerBound, (param.ctrl.nControlHorizon,1)).flatten()
        #print(f"lb: {lb}")
        ub = np.tile(param.ctrl.uUpperBound, (param.ctrl.nControlHorizon,1)).flatten()

        if param.useGradientOpt:
            wrapper = MPCResidualWithJacobian(t1, x1, u0, param)

            res = least_squares(wrapper.fun, U, jac=wrapper.jac, bounds=(lb, ub), **options)
        else:
            err = lambda U: errorMPC(t1, x1, u0, U, param)
            res = least_squares(err, U, bounds=(lb, ub), **options)
        
        #res = least_squares(err, U, bounds=(lb, ub), **options)
        U = res.x

    elif scenario == 'dock':
        options = {
            'method': 'SLSQP',
            'options': {
                'maxiter': 1000,
                'disp': True
            }
        }
        cost = lambda U: costMPC(t1, x1, u0, U, param)
        lb = np.tile(param.ctrl.uLowerBound, param.ctrl.nControlHorizon)
        ub = np.tile(param.ctrl.uUpperBound, param.ctrl.nControlHorizon)
        nonlcon = lambda U: nonlconMPC(t1, x1, u0, U, param)
        
        res = minimize(cost, U, bounds=np.column_stack((lb, ub)), constraints={'type': 'eq', 'fun': nonlcon}, **options)
        U = res.x

    else:
        raise ValueError("Unknown mode")

    return U