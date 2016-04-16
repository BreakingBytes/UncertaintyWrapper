.. _api:

API
===

Uncertainty Wrapper
-------------------

.. automodule:: uncertainty_wrapper

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

.. automodule:: uncertainty_wrapper.tests

Test Uncertainty Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: test_unc_wrapper

Test IV curve
~~~~~~~~~~~~~

.. autofunction:: test_IV

Test Solar Position
~~~~~~~~~~~~~~~~~~~

.. autofunction:: test_solpos
