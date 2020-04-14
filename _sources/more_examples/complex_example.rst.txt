.. _complex-example:

Complex Example
---------------

A more complex example from the :mod:`~uncertainty_wrapper.tests.test_uncertainty_wrapper`
module called :func:`~uncertainty_wrapper.tests.test_uncertainty_wrapper.test_IV`,
includes combinations of several exponential and power operations. It contains
9 input arguments, there 126 observations of each corresponding to different
voltages and there are 3 return values. The calculated uncertainty using a 1%
standard deviation (square root of variance) for all 9 inputs is shown below.

.. image:: /_static/IV_and_PV_plots_with_uncertainty.png

The test compares the derivatives calculated using central finite difference
approximation with an analytical calculation from 0.3[V] to 0.6[V]. Below 0.3[V]
the approximations deviate from the analytical for
:math:`\frac{\partial I_{sc}}{\partial I_{sat_{1,0}}}`,
:math:`\frac{\partial I_{sc}}{\partial I_{sat_2}}` and
:math:`\frac{\partial I_{sc}}{\partial E_g}` while all other independent
variables are consistently below 10e-7. The analytical derivatives are propagated
using `AlgoPy <https://pythonhosted.org/algopy/>`_, an automatic differentiation
package, which requires rewriting all NumPy operations like :math:`exp` using
AlgoPy. This makes it impractical for use in most models, but still useful for
testing.

.. image:: /_static/IV-PV-jac-errors.png
