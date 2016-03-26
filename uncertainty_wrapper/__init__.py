from functools import wraps, partial
import numpy as np
import numdifftools as nd  # 800ms per loop
import numdifftools.nd_algopy as nda  # 170us per loop, about 5000x faster!
import logging

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)
EPS = np.finfo(float).eps  # machine precision
DELTA = EPS ** 0.3333333333333333  # finite difference delta


def jacobian(x, func):
    """
    estimate Jacobian
    """
    nx = len(x)  # degrees of freedom
    x = np.array(x)
    LOGGER.debug(x)
    nnx = x[0].size
    assert all(nnx == _.size for _ in x)
    j = None  # matrix of zeros
    delta = np.eye(nx) * DELTA
    LOGGER.debug(delta)
    for d in delta:
        df = np.array(func(*(x.T + d).T)) - np.array(func(*(x.T - d).T))
        df = np.reshape(df / DELTA / 2, (-1, 1))
        if j is None:
            j = df
            LOGGER.debug(j)
        else:
            j = np.append(j, df, axis=1)  # derivatives df/d_n
    return j

#TODO: fix so that j is grouped correctly

# def uncertainty(f, x, dx, j=None):
#     """
#     estimate uncertainty
#     """
#     if j is None:
#         j = partial(jacobian(x, func=f)
#     #nx = len(x)  # degrees of freedom
#     #if len(dx) == nx:
#     #    dx *= np.eye(nx)
#     #return np.dot(np.dot(J, dx), J.T).diagonal()
#     return np.dot(j*j, dx)

def uncertainty(dx):
    def unc_wrapper(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            nargs = len(args)
            dt = np.dtype([('data', 'float', nargs), ('unc', 'float', nargs)])
            data = f(*args, **kwargs)
            j = partial(jacobian(args, func=f))
            unc = np.dot(j*j, dx)
            return np.array(zip(data, unc), dtype=dt)
        return wrapper
    return unc_wrapper
