.. image:: https://travis-ci.org/SunPower/UncertaintyWrapper.svg?branch=master
    :target: https://travis-ci.org/SunPower/UncertaintyWrapper
    
UncertaintyWrapper
==================

Use ``@unc_wrapper`` decorator to wrap any Python callable to append the
covariance and Jacobian matrices to the return values. See documentation and
tests for usage and examples.

Installation
------------

Use ``pip install UncertaintyWrapper`` to install from
`PyPI <https://pypi.python.org/pypi/UncertaintyWrapper>`_ or download a source
distribution, extract and use ``python setup.py install``.

Requirements
------------

* `NumPy <http://www.numpy.org/>`_

Optional Requirements
~~~~~~~~~~~~~~~~~~~~~

* `Nose <https://nose.readthedocs.org/en/latest/index.html>`_ for testing.
* `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ to build documentation.
* `NREL SOLPOS <http://rredc.nrel.gov/solar/codesandalgorithms/solpos/>`_ for testing
* `AlgoPy <https://pythonhosted.org/algopy/>`_ for testing

Usage
-----

Example::

    from uncertainty_wrapper import unc_wraper
    import numpy as np

    @unc_wrapper
    def f(x):
        return np.exp(x)

    x, cov = np.array([[1.0]]), np.array([[0.1]])
    f(x, __covariance__=cov)

Returns::

    (array([[ 2.71828183]]),      # exp(1.0)
     array([[[ 0.73890561]]]),    # (delta-f)^2 = (df/dx)^2 * (delta-x)^2
     array([[[ 2.71828183]]]))    # df/dx = exp(x)


History
-------
Releases are named after
`geological eons, periods and epochs <https://en.wikipedia.org/wiki/Geologic_time_scale>`_.

`v0.4.1 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.4.1>`_ `Paleozoic Era <https://en.wikipedia.org/wiki/Paleozoic>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Jagged arrays of covariance keys work now.
* simplify

`v0.4 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.4>`_ `Phanerozoic Era <https://en.wikipedia.org/wiki/Phanerozoic>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fixes #5, ``ValueError`` if covariance keys have multiple observations
* fix covariance cross terms not scaled correctly

`v0.3.3 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.3.3>`_ `Neoproterozoic Era <https://en.wikipedia.org/wiki/Neoproterozoic>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fixes #4, ``ValueError`` if just one observation

`v0.3.2 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.3.2>`_ `Mesoproterozoic Era <https://en.wikipedia.org/wiki/Mesoproterozoic>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fixes #2, don't need to tile scalar x for multiple observations
* Fixes #3, use sparse matrices for dot product instead of dense
* uses pvlib example instead of proprietary solar_utils


`v0.3.1 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.3.1>`_ `Paleoproterozoic Era <https://en.wikipedia.org/wiki/Paleoproterozoic>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixes #1 works with Pint's @ureg.wraps()
* Use indices for positional arguments. Don't use inspect.argspec since not
  guaranteed to be the same for wrapped or decorated functions
* Test Jacobian estimate for IV with `AlgoPy <https://pythonhosted.org/algopy/>`_
* Show Jacobian errors plot in getting started docs.


`v0.3 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.3>`_ `Proterozoic Eon <https://en.wikipedia.org/wiki/Proterozoic>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* new ``unc_wrapper_args()`` allows selection of independent variables that the
  partial derivatives are with respect to and also grouping those arguments
  together so that in the original function they can stay unpacked.
* return values are grouped correctly so that they can remain unpacked in
  original function. These allow Uncertainty Wrapper to be used with
  `Pint's wrapper <http://pint.readthedocs.org/en/latest/wrapping.html>`_
* covariance now specified as dimensionaless fraction of square of arguments
* more complex tests: IV curve and solar position (requires
  `NREL's solpos <http://rredc.nrel.gov/solar/codesandalgorithms/solpos/>`_)


`v0.2.1 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.2>`_ `Eoarchean Era <https://en.wikipedia.org/wiki/Eoarchean>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* update documentation


`v0.2 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.2>`_ `Archean Eon <https://en.wikipedia.org/wiki/Archean>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fix nargs and nf order mixup in Jacobian
* add more complex test
* fix tile cov by nobs
* move partial derivative to subfunction
* try threading, but same speed, and would only work with NumPy anyway


`v0.1 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.1>`_ `Hadean Eon <https://en.wikipedia.org/wiki/Hadean>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* adds covariance to output
* allows __covariance__ to be passed as input
* uses estimate Jacobian based on central finite difference method
