UncertaintyWrapper
==================

Use ``@unc_wrapper`` decorator to wrap any Python callable to append the
covariance and Jacobian matrices to the return values. See documentation and
tests for usage and examples.

Installation
------------

Use ``pip install uncertainty_wrapper`` to install from
`PyPI <https://pypi.python.org/pypi/uncertainty_wrapper>`_ or download a source
distribution, extract and use ``python setup.py install``.

Requirements
------------

* `NumPy <http://www.numpy.org/>`_
* `Nose <https://nose.readthedocs.org/en/latest/index.html>`_ for testing.
* `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ to build documentation.

History
-------

`v0.2 <https://github.com/SunPower/UncertaintyWrapper/releases/tag/v0.2>`_ `Archean Eon <https://en.wikipedia.org/wiki/Archean>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

Releases are named after
`geological eons, periods and epochs <https://en.wikipedia.org/wiki/Geologic_time_scale>`_.

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
