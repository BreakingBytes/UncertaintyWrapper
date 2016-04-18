.. _api:

API
===

Uncertainty Wrapper
-------------------

.. automodule:: uncertainty_wrapper.core

Step Size
~~~~~~~~~

.. autodata:: DELTA

Partial Derivative
~~~~~~~~~~~~~~~~~~

.. autofunction:: partial_derivative

Jacobian
~~~~~~~~

.. autofunction:: jacobian

Flatten Jacobian
~~~~~~~~~~~~~~~~

.. autofunction:: jflatten

Wrapper
~~~~~~~

.. autofunction:: unc_wrapper_args


Wrapper Shortcut
````````````````

.. function:: unc_wrapper

This is basically :func:`unc_wrapper_args()` with no argument which assumes that
all independent arguments are already grouped together.

Tests
-----

.. automodule:: uncertainty_wrapper.tests.test_uncertainty_wrapper

Test Uncertainty Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: test_unc_wrapper

Test simple exponential function with known derivative. Assert derivative is
correct and that uncertainty is propagated correctly.

Test IV curve
~~~~~~~~~~~~~

.. autofunction:: test_IV

Test complex function with several operations, including :math:`exp`, :math:`log`
and powers, with several input arguments and with several return values. Check
Jacobian calculated with central finite difference approximation with automatic
differentiation using AlgoPy.

Test Solar Position
~~~~~~~~~~~~~~~~~~~

.. autofunction:: test_solpos

Test function from a Python C/C++ extension. Check calcuated Jacobian with John
D'Errico's ``numdifftools`` Python package (MATLAB ``derivest``).
