.. _python-extension-example-with-units:

Python Extension Example with Units
-----------------------------------

Often Python packages contain extensions in C/C++ which can't be tested using
automatic differentiation. The Numdidfftools is an alternative package that can
calculate derivatives more accurately than the central finite difference
approximation.

An example of using :func:`~uncertainty_wrapper.core.unc_wrapper_args` with a
Python extension written in C/C++ is in the tests called
:func:`~uncertainty_wrapper.tests.test_uncertainty_wrapper.test_solpos`. This
test using C/C++ code from NREL that is called using Python ``ctypes`` module.

This example also demonstrates using Pint's units wrapper. When using the units
wrapper, you must use :func:`~uncertainty_wrapper.core.unc_wrapper_args` and
specify the indices of the positional arguments which corresond to the covariance
matrix. Also, two additional ``None, None`` should be appended to the units
wrapper return values because otherwise Pint uses ``zip(out_units, retvals)``
and therefore the covariance and Jacobian matrices will get dropped. ::

    @UREG.wraps(('deg', 'deg', 'dimensionless', 'dimensionless', None, None),
                ('deg', 'deg', 'millibar', 'degC', None, 'second'))
    @unc_wrapper_args(0, 1, 2, 3, 5)
    def solar_position(lat, lon, press, tamb, timestamps, seconds=0):
        pass

Then test it out. ::

    dt = PST.localize(datetime(2016, 4, 13, 12, 30, 0))
    lat = 37.405 * UREG.deg
    lon = -121.95 * UREG.deg
    press = 101325 * UREG.Pa
    tamb = 293.15 * UREG.degK
    seconds = 1 * UREG.s
    cov = np.diag([0.0001] * 5)
    ze, az, am, ampress, cov, jac = solar_position(lat, lon, press, tamb, dt,
                                                   seconds, __covariance__=cov)

The results are::

    # <Quantity([ 28.39483643], 'deg')>  # zenith
    # <Quantity([ 191.40260315], 'deg')>  # azimuth
    # <Quantity([ 1.1361022], 'dimensionless')>  # air mass
    # <Quantity([ 1.13638258], 'dimensionless')>  # pressure corrected air mass
    # covariance
    # array([[  1.34817971e+00,   1.58771853e+01,   1.44602315e-02,
                1.44602315e-02],
             [  1.58771853e+01,   2.09001667e+02,   1.70204117e-01,
                1.70204117e-01],
             [  1.44602315e-02,   1.70204117e-01,   1.55097144e-04,
                1.55097144e-04],
             [  1.44602315e-02,   1.70204117e-01,   1.55097144e-04,
                2.85468656e-04]])
    # Jacobian
    # array([[  9.76813540e-01,   1.65303295e-01,   0.00000000e+00,
                0.00000000e+00,   1.08353210e+02],
             [ -4.04198706e-01,   2.16960574e+00,   0.00000000e+00,
                0.00000000e+00,   1.42119094e+03],
             [  1.05260080e-02,   1.77571899e-03,   0.00000000e+00,
                0.00000000e+00,   1.16148972e+00],
             [  1.05260080e-02,   1.77571899e-03,   1.12687239e-03,
                0.00000000e+00,   1.16148972e+00]]))

Note that Pint corrects the ambient temperature from Kelvin to Celsius and also
converted Pascals to millibar. Finally Pint appends the specified units to the
return values.
