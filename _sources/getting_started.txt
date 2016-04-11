.. _getting-started:

Getting Started
===============
You can wrap any Python callable using :func:`~uncertainty_wrapper.unc_wrapper`
that adds covariance as an input argument then calculates and appends the
covariance and Jacobian matrices to the return values. However you will need to
manipulate the input arguments to match the expected :ref:`API`.

Example
-------
The following examples are from `Uncertainty Benchmarks <https://github.com/mikofski/uncertainty_benchmarks>`_::

    from uncertainty_wrapper import unc_wrapper
    # unc_wrapper expects input args to be 2-D NumPy arrays
    import numpy as np

    # simple test functions with multiple input arguments and return
    # values and whose derivatives are easily derived.
    NARGS = 2  # number of input arguments
    F = lambda x: np.array([x[0] ** 2 + 2 * x[0] * x[1] + x[1] ** 2,
                            x[1] ** 2 - x[0] ** 2])
    G = lambda x: np.array([(2 * (x[0] + x[1]), 2 * (x[1] + x[0])),
                            (-2 * x[0], 2 * x[1])])
    AVG = np.random.rand(NARGS) * 10.  # some test input arguments
    # randomly generated symmetrical covariance matrix
    COV = np.random.rand(NARGS, NARGS) / 10.
    TOL = 1e-6  # make it sparse, set anything less than 1e-6 to zero
    COV = np.where(COV > TOL, COV, np.zeros((NARGS, NARGS)))
    COV *= COV.T  # must be symmetrical

    @unc_wrapper
    def example(avg=AVG, f=F):
        """Example of unc_wrapper usage"""
        avg = f(avg)
        return avg

    # uses @wraps from functools so docstrings should work
    print example.__doc__
    # Example of unc_wrapper usage

    # reshape args as row stack since there is only one observation and
    # unc_wrapper expects there to be multiple observations
    AVG = AVG.reshape((NARGS, 1))

    # the wrapped example now takes a second argument called
    # __covariance__
    retval = example(AVG, COV, F)
    # and appends covariance and Jacobian matrices to the return values
    avg, cov, jac = retval

    # squeeze out extra dimension we added since there's only one
    # observation and display results
    avg = avg.squeeze()
    print avg
    # [ 11.13976423,   3.27799112]
    print cov
    # [[ 0.27347267,  0.13844758],
    #  [ 0.13844758,  0.07171001]])
    print jac
    # [[ 6.67525707,  6.67525707],
    #  [-2.35549673,  4.31976033]])

    # compare to analytical derivatives
    print G(AVG).squeeze()
    # [[ 6.67525707  6.67525707]
    #  [-2.35549673  4.31976033]]
