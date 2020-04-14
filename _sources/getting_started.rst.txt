.. _getting-started:

Getting Started
===============
You can wrap any Python callable using :func:`~uncertainty_wrapper.core.unc_wrapper`
or :func:`~uncertainty_wrapper.core.unc_wrapper_args`, that does the following:

* looks for ``__covariance__`` as a keyword argument
* calculates the Jacobian and covariance matrices
* appends the Jacobian and covariance matrices to the return values.

However you may need to manipulate the input arguments to match the expected
:ref:`API`.

Simple Example
--------------

This simple example using two input arguments and two return values is from
`Uncertainty Benchmarks <https://github.com/mikofski/uncertainty_benchmarks>`_::

    from uncertainty_wrapper import unc_wrapper
    # unc_wrapper expects input args to be 2-D NumPy arrays
    import numpy as np

    # simple test functions with multiple input arguments and return
    # values and whose derivatives are easily derived.
    NARGS = 2  # number of input arguments
    F = lambda x: np.array([(x[0] + x[1]) ** 2, x[1] ** 2 - x[0] ** 2])
    G = lambda x: np.array([(2 * (x[0] + x[1]), 2 * (x[1] + x[0])),
                            (-2 * x[0], 2 * x[1])])
    AVG = np.random.rand(NARGS) * 10.  # some test input arguments
    # randomly generated symmetrical covariance matrix
    COV = np.random.rand(NARGS, NARGS) / 10.
    COV = (COV + COV.T) / 2.0  # must be symmetrical

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
    print AVG
    # [[ 1.80222955]
    #  [ 5.62897685]]

    # the wrapped example now takes a second argument called
    # __covariance__
    print COV
    # [[ 0.06798386  0.05971218]
    #  [ 0.05971218  0.09359305]]

    retval = example(AVG, F, __covariance__=COV)
    # and appends covariance and Jacobian matrices to the return values
    avg, cov, jac = retval

    # squeeze out extra dimension we added since there's only one
    # observation and display results
    avg = avg.squeeze()
    print avg
    # [ 55.22282851  28.43734901]

    print cov
    # [[ 1164.60425675   790.5452895 ]
    #  [  415.45944116   294.07938566]]

    print jac
    # [[ 14.86241279  14.86241279]
    #  [ -3.6044591   11.2579537 ]]

    # compare to analytical derivatives
    print G(AVG).squeeze()
    # [[ 14.86241279  14.86241279]
    #  [ -3.6044591   11.2579537 ]]

More Examples
-------------

* :ref:`complex-example`
* :ref:`python-extension-example-with-units`

The next sections contain more examples cover more advanced usage. Uncertanty
Wrapper can consider multiple inputs arguments and return values. It can also be
used with Python extensions written in c/c++. Finally
:func:`~uncertainty_wrapper.core.unc_wrapper_args` can be used to specify which
args are indepndent to include in the covariance and Jaconbian.

Announcement
------------
Previous versions of Uncertainty Wrapper have worked with Pint's units wrapper
to automatically check units, but unfortunately this is no longer supported.
