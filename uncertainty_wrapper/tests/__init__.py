"""tests"""

from nose.tools import ok_, eq_, raises
import numpy as np
from uncertainty_wrapper import unc_wrapper
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def test_unc_wrapper():
    """
    test uncertainty wrapper
    """
    x, cov = np.array([[1.0]]), np.array([[0.1]])
    
    @unc_wrapper
    def f(y):
        return np.exp(y)
    
    avg, var = f(x, __covariance__=cov)
    LOGGER.debug("average = %g", avg)
    LOGGER.debug("variance = %g", var)
    ok_(np.isclose(avg, np.exp(x)))
    ok_(np.isclose(var, cov * np.exp(x) ** 2))
    return avg, var


if __name__ == '__main__':
    test_unc_wrapper()
