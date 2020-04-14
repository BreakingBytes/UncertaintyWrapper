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

This example also demonstrates calculating covariance for multiple
observations using :func:`~uncertainty_wrapper.core.unc_wrapper_args` to
specify the indices of the positional arguments which corresond to the
covariance matrix.::

    @unc_wrapper_args(1, 2, 3, 4, 5)
    # indices specify positions of independent variables:
    # 1: latitude, 2: longitude, 3: pressure, 4: altitude, 5: temperature
    def spa(times, latitude, longitude, pressure, altitude, temperature):
        dataframe = pvlib.solarposition.spa_c(times, latitude, longitude, pressure,
                                              temperature)
        retvals = dataframe.to_records()
        zenith = retvals['apparent_zenith']
        zenith = np.where(zenith < 90, zenith, np.nan)
        azimuth = retvals['azimuth']
        return zenith, azimuth

Then test it out. ::

    times = pd.DatetimeIndex(pd.date_range(
        start='2015/1/1', end='2015/1/2', freq='1h', tz=PST)).tz_convert(UTC)
    latitude, longitude = 37.0, -122.0  # degrees
    pressure, temperature = 101325.0, 22.0  # Pa, degC
    altitude = 0.0
    # standard deviation of 1% assuming normal distribution
    covariance = np.diag([0.0001] * 5)
    ze, az, cov, jac = spa(times, latitude, longitude, pressure, altitude,
                           temperature, __covariance__=covariance)

The results are::

    >>> ze
    <ndarray([         nan          nan          nan          nan          nan
              nan          nan          nan  84.10855021  74.98258957
      67.47442104  62.27279883  60.00799371  61.01651321  65.14311785
      71.83729124  80.41979434  89.92923993          nan          nan
              nan          nan          nan          nan          nan],
    dtype='float')>

    >>> az
    <ndarray([ 349.29771499   40.21062767   66.71930375   80.93018543   90.85288686
       99.21242575  107.18121735  115.45045069  124.56418347  135.02313717
      147.24740279  161.37157806  176.92280365  192.74232655  207.51976817
      220.49410796  231.60091006  241.18407504  249.7263611   257.75154961
      265.87317048  275.01453439  287.07887655  307.28364551  348.92138471],
    dtype='float')>

Note: previous versions of uncertainty wrapper worked with Pint to check
units, but unfortunatley this is no longer supported. ::

    >>> idx = 8  # covariance at 8AM
    >>> times[idx]
    Timestamp('2015-01-01 08:00:00-0800', tz='US/Pacific', offset='H')
    
    >>> nf = 2  # number of dependent variables: [ze, az]
    >>> print cov[(nf * idx):(nf * (idx + 1)), (nf * idx):(nf * (idx + 1))]
    [[ 0.66082282, -0.61487518],
     [-0.61487518,  0.62483904]]

    >>> print np.sqrt(cov[(nf * idx), (nf * idx)]) / ze[idx]  # standard deviation
    0.0096710802029002577

This tells us that the standard deviation of the zenith is 1% if the input has a standard deviation
of 1%. That's reasonable.

    >>> nargs = 5  # number of independent args
    >>> jac[nf*(idx-1):nf*idx, nargs*(idx-1):nargs*idx]  # Jacobian at 8AM
    [[  5.56075143e-01,  -6.44623321e-01,  -1.42364184e-06, 0.00000000e+00,   1.06672083e-10],
     [  8.29163154e-02,   6.47436098e-01,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00]]

This also tells that zenith is more sensitive to latitude and longitude than pressure or temperature
and more sensitive to latitude than azimuth is.

Perhaps the most interesting outcome is the negative covariance between Zenith and Azimuth. From
`Wikipedia <https://en.wikipedia.org/wiki/Covariance>`_

    ... when the greater values of one variable mainly correspond to the lesser values of the other,
    the covariance is negative.

In other words when the error in Zenith increases, the error in Azimuth decreases. This is not
uncommon but it's not always intuitively obvious; we generally think that to get the largest error
we should choose the largest errors for all independent variables.

