"""
Uncertainty wrapper calculates uncertainties of wrapped functions using
central finite difference approximation of the Jacobian matrix.

.. math::

    dF_{ij} = J_{ij} * S_{x_i}{x_j} * J_{ij}^{T}

Diagonals of :math:`dF_{ij}` are standard deviations squared.

SunPower Corp. (c) 2016
"""

from functools import wraps
import numpy as np
import inspect
import logging

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
__VERSION__ = '0.2.1'
__RELEASE__ = u"Eoarchean Era"
__URL__ = u'https://github.com/SunPower/UncertaintyWrapper'
__AUTHOR__ = u"Mark Mikofski"
__EMAIL__ = u'mark.mikofski@sunpowercorp.com'

DELTA = np.finfo(float).eps ** (1.0 / 3.0) / 2.0


def partial_derivative(f, x, n, nargs, nobs, delta=DELTA):
    """
    Calculate partial derivative using central finite difference approximation.

    :param f: function
    :param x: sequence of arguments
    :param n: index of argument derivateve is with respect to
    :param nargs: number of arguments
    :param nobs: number of observations
    :param delta: optional step size, default is :math:`\\epsilon^{1/3}` where
        :math:`\\epsilon` is machine precision
    """
    dx = np.zeros((nargs, nobs))
    # scale delta by (|x| + 1.0) to avoid noise from machine precision
    dx[n] += np.where(x[n], x[n] * delta, delta)
    # apply central difference approximation
    return (f(x + dx) - f(x - dx)) / dx[n] / 2.0


# TODO: make this a class, add DELTA as class variable and flatten as method
def jacobian(func, x, *args, **kwargs):
    """
    Estimate Jacobian matrices :math:`\\frac{\\partial f_i}{\\partial x_{j,k}}`
    where :math:`k` are independent observations of :math:`x`.

    The independent variable, :math:`x`, must be a numpy array with exactly 2
    dimensions. The first dimension is the number of independent arguments,
    and the second dimensions is the number of observations.

    The function must return a Numpy array with exactly 2 dimensions. The first
    is the number of returns and the second dimension corresponds to the number
    of observations. If the input argument is 2-D then the output should also
    be 2-D

    Constant arguments can be passed as additional positional arguments or
    keyword arguments. If any constant argument increases the number of
    observations of the return value, tile the input arguments to match.

    Use :func:`numpy.atleast_2d` or :func:`numpy.reshape` to get the
    correct dimensions for scalars.

    :param func: function
    :param x: independent variables grouped by observation
    :return: Jacobian matrices for each observation
    """
    nargs = x.shape[0]  # degrees of freedom
    nobs = x.size / nargs  # number of observations
    f = lambda x_: func(x_, *args, **kwargs)
    j = None  # matrix of zeros
    for n in xrange(nargs):
        df = partial_derivative(f, x, n, nargs, nobs)
        if j is None:
            j = np.zeros((nargs, df.shape[0], nobs))
        j[n] = df
        # better to transpose J once than to transpose df each time
        # j[:,:,n] = df.T
    return j.T


def jflatten(j):
    """
    Flatten Jacobian into 2-D
    """
    nobs, nf, nargs = j.shape
    nrows, ncols = nf * nobs, nargs * nobs
    jflat = np.zeros((nrows, ncols))
    for n in xrange(nobs):
        r, c = n * nf, n * nargs
        jflat[r:(r + nf), c:(c + nargs)] = j[n]
    return jflat


# TODO: allow user to supply analytical Jacobian, only fall back on Jacob
# estimate if jac is None
# TODO: check for negative covariance, what do we do?
# TODO: what is the expected format for COV if some have multiple
# observations, is it necessary to flatten J first??
# group args as specified


def unc_wrapper_args(*covariance_keys):
    def unc_wrapper(f):
        """
        Wrap function, pop ``__covariance__`` argument from keyword arguments,
        propagate uncertainty given covariance using Jacobian estimate. and append
        calculated covariance and Jacobian matrices to return values.
        """
        argspec = inspect.getargspec(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            cov_keys = covariance_keys
            cov = kwargs.pop('__covariance__', None)  # pop covariance
            if argspec.defaults is not None:
                ndflts = len(argspec.defaults)
                kwargs.update(zip(argspec.args[-ndflts:], argspec.defaults))
            kwargs.update(zip(argspec.args, args))  # convert args to kwargs
            if len(cov_keys) > 0:
                x = np.array([np.atleast_1d(kwargs.pop(k)) for k in cov_keys])
            elif cov_keys is None:
                cov_keys = kwargs.keys()
                x = np.reshape(kwargs.values(), (len(cov_keys), -1))
            else:
                x = kwargs.pop(argspec.args[0])

            def g(y, **gkwargs):
                if cov_keys:
                    gkwargs.update(zip(cov_keys, y))
                    return np.array(f(**gkwargs))
                # assumes independent and dependent vars already grouped
                return f(y, **gkwargs)

            avg = g(x, **kwargs)
            jac = jacobian(g, x, **kwargs)
            # covariance must account for all observations
            if cov is not None and cov.ndim == 3:
                # if covariance is an array of covariances, flatten it.
                cov = jflatten(cov)
            jac = jflatten(jac)
            if cov is not None:
                cov = np.dot(np.dot(jac, cov * x.T.flatten()), jac.T)
            return avg, cov, jac
        return wrapper
    return unc_wrapper

unc_wrapper = unc_wrapper_args()
