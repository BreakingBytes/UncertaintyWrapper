"""
Uncertainty wrapper calculates uncertainties of wrapped functions using
central finite difference approximation of the Jacobian matrix.

.. math::

    \\frac{\\partial f_i}{\\partial x_{j,k}}

Uncertainty of the output is propagated using first order terms of a Taylor
series expansion around :math:`x`.


.. math::

    dF_{ij} = J_{ij} * S_{x_i, x_j} * J_{ij}^{T}

Diagonals of :math:`dF_{ij}` are standard deviations squared.

SunPower Corp. (c) 2016
"""
from __future__ import division

from builtins import zip
from builtins import range
from past.builtins import basestring
from past.utils import old_div
from functools import wraps
import numpy as np
import logging
from multiprocessing import Pool
from scipy.sparse import csr_matrix

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
DELTA = np.finfo(float).eps ** (1.0 / 3.0) / 2.0


def prop_unc(jc):
    """
    Propagate uncertainty.

    :param jc: the Jacobian and covariance matrix
    :type jc: sequence

    This method is mainly designed to be used as the target for a
    multiprocessing pool.
    """
    j, c = jc
    return np.dot(np.dot(j, c), j.T)


def partial_derivative(f, x, n, nargs, delta=DELTA):
    """
    Calculate partial derivative using central finite difference approximation.

    :param f: function
    :param x: sequence of arguments
    :param n: index of argument derivateve is with respect to
    :param nargs: number of arguments
    :param delta: optional step size, default is :math:`\\epsilon^{1/3}` where
        :math:`\\epsilon` is machine precision
    """
    dx = np.zeros((nargs, len(x[n])))
    # scale delta by (|x| + 1.0) to avoid noise from machine precision
    dx[n] += np.where(x[n], x[n] * delta, delta)
    # apply central difference approximation
    x_dx = list(zip(*[xi + (dxi, -dxi) for xi, dxi in zip(x, dx)]))
    return old_div((f(x_dx[0]) - f(x_dx[1])), dx[n]) / 2.0


# TODO: make this a class, add DELTA as class variable and flatten as method
def jacobian(func, x, nf, nobs, *args, **kwargs):
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
    :param nf: number of return in output (1st dimension)
    :param nobs: number of observations in output (2nd dimension)
    :return: Jacobian matrices for each observation
    """
    nargs = len(x)  # degrees of freedom
    f = lambda x_: func(x_, *args, **kwargs)
    j = np.zeros((nargs, nf, nobs))  # matrix of zeros
    for n in range(nargs):
        j[n] = partial_derivative(f, x, n, nargs)
        # better to transpose J once than transpose partial derivative each time
        # j[:,:,n] = df.T
    return j.T


def jflatten(j):
    """
    Flatten 3_D Jacobian into 2-D.
    """
    nobs, nf, nargs = j.shape
    nrows, ncols = nf * nobs, nargs * nobs
    jflat = np.zeros((nrows, ncols))
    for n in range(nobs):
        r, c = n * nf, n * nargs
        jflat[r:(r + nf), c:(c + nargs)] = j[n]
    return jflat


def jtosparse(j):
    """
    Generate sparse matrix coordinates from 3-D Jacobian.
    """
    data = j.flatten().tolist()
    nobs, nf, nargs = j.shape
    indices = list(zip(*[(r, c) for n in range(nobs)
                    for r in range(n * nf, (n + 1) * nf)
                    for c in range(n * nargs, (n + 1) * nargs)]))
    return csr_matrix((data, indices), shape=(nobs * nf, nobs * nargs))


# TODO: allow user to supply analytical Jacobian, only fall back on Jacob
# estimate if jac is None


def unc_wrapper_args(*covariance_keys):
    """
    Wrap function, calculate its Jacobian and calculate the covariance of the
    outputs given the covariance of the specified inputs.

    :param covariance_keys: indices and names of arguments corresponding to
        covariance
    :return: wrapped function bound to specified covariance keys

    This is the outer uncertainty wrapper that allows you to specify the
    arguments in the original function that correspond to the covariance. The
    inner wrapper takes the original function to be wrapped. ::

        def f(a, b, c, d, kw1='foo', *args, **kwargs):
            pass

        # arguments a, c, d and kw1 correspond to the covariance matrix
        f_wrapped = unc_wrapper_args(0, 2, 3, 'kw1')(f)

        cov = np.array([[0.0001, 0., 0., 0.], [0., 0.0001, 0., 0.],
                        [0., 0., 0.0001, 0.], [0., 0., 0., 0.0001])
        y, cov, jac = f_wrapped(a, b, c, d, kw1='bar', __covariance__=cov)

    The covariance keys can be indices of positional arguments or the names of
    keywords argument used in calling the function. If no covariance keys are
    specified then the arguments that correspond to the covariance shoud be
    grouped into a sequence. If ``None`` is anywhere in ``covariance_keys`` then
    all of the arguments will be used to calculate the Jacobian.

    The covariance matrix must be a symmetrical matrix with positive numbers on
    the diagonal that correspond to the square of the standard deviation, second
    moment around the mean or root-mean-square(RMS) of the function with respect
    to the arguments specified as covariance keys. The other elements are the
    covariances corresponding to the arguments intersecting at that element.
    Pass the covariance matrix with the keyword ``__covariance__`` and it will
    be popped from the dictionary of keyword arguments provided to the wrapped
    function.

    The wrapped function will return the evaluation of the original function,
    its Jacobian, which is the sensitivity of the return output to each
    argument specified as a covariance key and the covariance propagated using
    the first order terms of a Taylor series expansion around the arguments.

    An optional keyword argument ``__method__`` can also be passed to the
    wrapped function (not the wrapper) that specifies the method used to
    calculate the dot product. The default method is ``'loop'``. The other
    methods are ``'dense'``, ``'sparse'`` and ``'pool'``.

    If the arguments specified as covariance keys are arrays, they should all be
    the same size. These dimensions will be considered as separate observations.
    Another argument, not in the covariance keys, may also create observations.
    The resulting Jacobian will have dimensions of number of observations (nobs)
    by number of return output (nf) by number of covariance keys (nargs). The
    resulting covariance will be nobs x nf x nf.
    """
    def wrapper(f):
        @wraps(f)
        def wrapped_function(*args, **kwargs):
            cov = kwargs.pop('__covariance__', None)  # pop covariance
            method = kwargs.pop('__method__', 'loop')  # pop covariance
            # covariance keys cannot be defaults, they must be in args or kwargs
            cov_keys = covariance_keys
            # convert args to kwargs by index
            kwargs.update({n: v for n, v in enumerate(args)})
            args = ()  # empty args
            if None in cov_keys:
                # use all keys
                cov_keys = list(kwargs.keys())
            # group covariance keys
            if len(cov_keys) > 0:
                # uses specified keys
                x = [np.atleast_1d(kwargs.pop(k)) for k in cov_keys]
            else:
                # arguments already grouped
                x = kwargs.pop(0)  # use first argument
            # remaining args
            args_dict = {}

            def args_from_kwargs(kwargs_):
                """unpack positional arguments from keyword arguments"""
                # create mapping of positional arguments by index
                args_ = [(n, v) for n, v in kwargs_.items()
                         if not isinstance(n, basestring)]
                # sort positional arguments by index
                idx, args_ = list(zip(*sorted(args_, key=lambda m: m[0])))
                # remove args_ and their indices from kwargs_
                args_dict_ = {n: kwargs_.pop(n) for n in idx}
                return args_, args_dict_

            if kwargs:
                args, args_dict = args_from_kwargs(kwargs)

            def f_(x_, *args_, **kwargs_):
                """call original function with independent variables grouped"""
                args_dict_ = args_dict
                if cov_keys:
                    kwargs_.update(args_dict_)
                    kwargs_.update(list(zip(cov_keys, x_)))
                if kwargs_:
                    args_, _ = args_from_kwargs(kwargs_)
                    return np.array(f(*args_, **kwargs_))
                # assumes independent variables already grouped
                return f(x_, *args_, **kwargs_)

            # evaluate function and Jacobian
            avg = f_(x, *args, **kwargs)
            # number of returns and observations
            if avg.ndim > 1:
                nf, nobs = avg.shape
            else:
                nf, nobs = avg.size, 1
            jac = jacobian(f_, x, nf, nobs, *args, **kwargs)
            # calculate covariance
            if cov is not None:
                # covariance must account for all observations
                # scale covariances by x squared in each direction
                if cov.ndim == 3:
                    x = np.array([np.repeat(y, nobs) if len(y)==1
                                  else y for y in x])
                    LOGGER.debug('x:\n%r', x)
                    cov = np.array([c * y * np.row_stack(y)
                                    for c, y in zip(cov, x.T)])
                else: # x are all only one dimension
                    x = np.asarray(x)
                    cov = cov * x * x.T
                    assert old_div(old_div(jac.size, nf), nobs) == old_div(cov.size, len(x))
                    cov = np.tile(cov, (nobs, 1, 1))
                # propagate uncertainty using different methods
                if method.lower() == 'dense':
                    j, c = jflatten(jac), jflatten(cov)
                    cov = prop_unc((j, c))
                # sparse
                elif method.lower() == 'sparse':
                    j, c = jtosparse(jac), jtosparse(cov)
                    cov = j.dot(c).dot(j.transpose())
                    cov = cov.todense()
                # pool
                elif method.lower() == 'pool':
                    try:
                        p = Pool()
                        cov = np.array(p.map(prop_unc, list(zip(jac, cov))))
                    finally:
                        p.terminate()
                # loop is the default
                else:
                    cov = np.array([prop_unc((jac[o], cov[o]))
                                    for o in range(nobs)])
                # dense and spares are flattened, unravel them into 3-D list of
                # observations
                if method.lower() in ['dense', 'sparse']:
                    cov = np.array([
                        cov[(nf * o):(nf * (o + 1)), (nf * o):(nf * (o + 1))]
                        for o in range(nobs)
                    ])
            # unpack returns for original function with ungrouped arguments
            if None in cov_keys or len(cov_keys) > 0:
                return tuple(avg.tolist() + [cov, jac])
            # independent variables were already grouped
            return avg, cov, jac
        return wrapped_function
    return wrapper

# short cut for functions with arguments already grouped
unc_wrapper = unc_wrapper_args()
unc_wrapper.__doc__ = "equivalent to unc_wrapper_args() with no arguments"
unc_wrapper.__name__ = "unc_wrapper"
