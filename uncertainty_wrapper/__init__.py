"""
Uncertainty wrapper calculates uncertainties of wrapped functions using
ALGOPY.

.. math::

    dF_{ij} = J_{ij} * S_{x_i}{x_j} * J_{ij}^{-1}

Diagonals of :math:`dF_{ij}` are standard deviations squared.
"""

from functools import wraps, partial
import numpy as np
from algopy import zeros as azeros
import numdifftools.nd_algopy as nda  # 170us per loop, about 5000x faster!
import logging

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)


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

def uncertainty(dx, jac=None):
    def unc_wrapper(f):
        inner_jac = jac
        if inner_jac is None:
            inner_jac = nda.Jacobian(f)
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = f(*args, **kwargs)
            j = inner_jac(*args, **kwargs)
            unc = np.dot(j*j, dx)
            nargs = len(args)
            dt = np.dtype([('data', 'float', nargs), ('unc', 'float', nargs)])
            return np.array(zip(data, unc), dtype=dt)
        return wrapper
    return unc_wrapper




def unc_decorator(dx, jac=None):
    def outer_wrapper(f):
        if jac is None:
            inner_jac = nda.Jacobian(f)
        def inner_wrapper(*args, **kwargs):
            y = f(*args, **kwargs)
        return inner_wrapper
    return outer_wrapper
