"""
Tests for :func:`~uncertainty_wrapper.unc_wrapper` and
:func:`~uncertainty_wrapper.unc_wrapper_args`
"""
from __future__ import division

# import ok_, np, pd,unc_wrapper, unc_wrapper_args, KB, QE,
# pvlib, plt, UREG, PST and LOGGER from .tests
from builtins import range
from past.utils import old_div
from uncertainty_wrapper.tests import *
from uncertainty_wrapper.tests.test_algopy import IV_algopy_jac, solpos_nd_jac

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def test_unc_wrapper():
    """
    Test uncertainty wrapper with grouped arguments.
    """
    x, cov = np.array([[1.0]]), np.array([[0.1]])

    @unc_wrapper
    def f(y):
        return np.exp(y)

    avg, var, jac = f(x, __covariance__=cov, __method__='dense')
    LOGGER.debug("average = %g", avg)
    LOGGER.debug("variance = %g", var)
    ok_(np.isclose(avg, np.exp(x)))
    ok_(np.isclose(var, cov * np.exp(x) ** 2))
    ok_(np.isclose(jac, np.exp(x)))
    return avg, var, jac


def test_unc_wrapper_args():
    """
    Test uncertainty wrapper with ungrouped arguments.
    """
    x, cov = 1.0, np.array([[0.1]])

    @unc_wrapper_args(None)
    def f(y):
        return np.exp(y)

    avg, var, jac = f(x, __covariance__=cov, __method__='dense')
    LOGGER.debug("average = %g", avg)
    LOGGER.debug("variance = %g", var)
    ok_(np.isclose(avg, np.exp(x)))
    ok_(np.isclose(var, cov * np.exp(x) ** 2))
    ok_(np.isclose(jac, np.exp(x)))
    return avg, var, jac


def test_multiple_observations():
    """
    Test multiple observations.
    """
    x, cov = [1.0, 1.0], np.array([[[0.1]], [[0.1]]])

    @unc_wrapper_args(None)
    def f(y):
        return np.exp(y).reshape(1, -1)

    avg, var, jac = f(x, __covariance__=cov, __method__='dense')
    LOGGER.debug("average = %r", avg)
    LOGGER.debug("variance = %r", var)
    ok_(np.allclose(avg, np.exp(x)))
    ok_(np.allclose(var, cov * np.exp(x) ** 2))
    ok_(np.allclose(jac, np.exp(x)))
    return avg, var, jac


def test_jagged():
    """
    Test jagged array.
    """
    w, x = 1.0, [1.0, 1.0]
    cov = np.array([[[0.1, 0.], [0., 0.1]],
                    [[0.1, 0.], [0., 0.1]]])

    @unc_wrapper_args(None)
    def f(y, z):
        return (y * np.exp(z)).reshape(1, -1)

    avg, var, jac = f(w, x, __covariance__=cov, __method__='dense')
    LOGGER.debug("average = %r", avg)
    LOGGER.debug("jacobian = %r", jac)
    LOGGER.debug("variance = %r", var)
    ok_(np.allclose(avg, w * np.exp(x)))
    ok_(np.allclose(jac, [np.exp(x), np.exp(x)]))
    var_calc = np.concatenate(
        [[np.exp(x) * cov[:, 0, 0] + np.exp(x) * cov[:, 1, 0]],
         [np.exp(x) * cov[:, 0, 1] + np.exp(x) * cov[:, 1, 1]]], 0
    ).reshape(2, 1, 2)
    var_calc = var_calc[:, 0, 0] * np.exp(x) + var_calc[:, 0, 1] * np.exp(x)
    ok_(np.allclose(var, var_calc))
    return avg, var, jac


def IV(x, Vd):
    """
    Calculate IV curve using 2-diode model.

    :param x: independent variables:
    :type x: sequence
    :param Vd: diode voltages
    :type Vd: :class:`numpy.ndarray`
    :returns: current [A], voltage [V] and power [W]
    :rtype: :class:`numpy.ndarray`

    The sequence of independent variables must contain the following in the
    specified order::

        [Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg]

    This function is an example of grouping the independent variables together
    so that :class:~`uncertianty_wrapper.core.unc_wrapper` can be used.
    """
    Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg = x
    Vt = old_div(Tc * KB, QE)
    Isc = Ee * Isc0 * (1.0 + (Tc - T0) * alpha_Isc)
    Isat1 = (
        Isat1_0 * (old_div(Tc ** 3.0, T0 ** 3.0)) *
        np.exp(old_div(Eg * QE, KB) * (1.0 / T0 - 1.0 / Tc))
    )
    Vd_sc = Isc * Rs  # at short circuit Vc = 0 
    Id1_sc = Isat1 * (np.exp(old_div(Vd_sc, Vt)) - 1.0)
    Id2_sc = Isat2 * (np.exp(Vd_sc / 2.0 / Vt) - 1.0)
    Ish_sc = old_div(Vd_sc, Rsh)
    Iph = Isc + Id1_sc + Id2_sc + Ish_sc
    Id1 = Isat1 * (np.exp(old_div(Vd, Vt)) - 1.0)
    Id2 = Isat2 * (np.exp(Vd / 2.0 / Vt) - 1.0)
    Ish = old_div(Vd, Rsh)
    Ic = Iph - Id1 - Id2 - Ish
    Vc = Vd - Ic * Rs
    return np.array([Ic, Vc, Ic * Vc])


def Voc(x):
    """
    Estimate open circuit voltage (Voc).
    """
    Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg = x
    msg = ['Ee=%g[suns]','Tc=%g[K]','Rs=%g[ohms]','Rsh=%g[ohms]',
           'Isat1_0=%g[A]','Isat2=%g[A]','Isc0=%g[A]','alpha_Isc=%g[]',
           'Eg=%g[eV]']
    LOGGER.debug('\n' + '\n'.join(msg) + '\n', *x)
    Vt = old_div(Tc * KB, QE)
    LOGGER.debug('Vt=%g[V]', Vt)
    Isc = Ee * Isc0 * (1.0 + (Tc - T0) * alpha_Isc)
    LOGGER.debug('Isc=%g[A]', Isc)
    Isat1 = (
        Isat1_0 * (old_div(Tc ** 3.0, T0 ** 3.0)) *
        np.exp(old_div(Eg * QE, KB) * (1.0 / T0 - 1.0 / Tc))
    )
    LOGGER.debug('Isat1=%g[A]', Isat1)
    Vd_sc = Isc * Rs  # at short circuit Vc = 0 
    Id1_sc = Isat1 * (np.exp(old_div(Vd_sc, Vt)) - 1.0)
    Id2_sc = Isat2 * (np.exp(Vd_sc / 2.0 / Vt) - 1.0)
    Ish_sc = old_div(Vd_sc, Rsh)
    Iph = Isc + Id1_sc + Id2_sc + Ish_sc
    # estimate Voc
    delta = Isat2 ** 2.0 + 4.0 * Isat1 * (Iph + Isat1 + Isat2)
    return Vt * np.log(((-Isat2 + np.sqrt(delta)) / 2.0 / Isat1) ** 2.0)


# constants for IV test
RS = 0.004267236774264931  # [ohm] series resistance
RSH = 10.01226369025448  # [ohm] shunt resistance
ISAT1_0 = 2.286188161253440E-11  # [A] diode one saturation current
ISAT2 = 1.117455042372326E-6  # [A] diode two saturation current
ISC0 = 6.3056  # [A] reference short circuit current
EE = 0.8  # [suns] effective irradiance
TC = 323.15  # [K] cell temperature
EG = 1.1  # [eV] c-Si band gap
ALPHA_ISC = 0.0003551  # [1/degC] short circuit current temp co 
# [V] open circuit voltage
VOC = Voc((EE, TC, RS, RSH, ISAT1_0, ISAT2, ISC0, ALPHA_ISC, EG))
assert np.isclose(VOC, 0.62816490891656673)
LOGGER.debug('Voc = %g[V]', VOC)
VD = np.arange(0, VOC, 0.005)  # [V] diode voltages
X = np.array([EE, TC, RS, RSH, ISAT1_0, ISAT2, ISC0, ALPHA_ISC, EG])
X = X.reshape(-1, 1)
# covariance equivalent to standard deviation of 1.0 [%]
COV = np.diag([1e-4] * X.size)
X_algopy = X.repeat(VD.size, axis=1)


def test_IV(method='sparse'):
    """
    Test calculation of photovoltaic cell IV curve using 2-diode model and
    and compare Jacobian estimated by finite central difference to AlgoPy
    automatic differentiation.
    """
    f = unc_wrapper(IV)
    pv, pv_cov, pv_jac = f(X, VD, __covariance__=COV, __method__=method)
    pv_cov = jflatten(pv_cov)
    pv_jac = jflatten(pv_jac)
    pv_jac_algopy = IV_algopy_jac(*X_algopy, Vd=VD)
    nVd = pv_jac_algopy.shape[1]
    for n in range(nVd // 2, nVd):
        irow, icol = 3 * n, 9 * n
        jrow, jcol = 3 + irow, 9 +icol
        pv_jac_n = pv_jac[irow:jrow, icol:jcol]
        pv_jac_algopy_n = pv_jac_algopy[:, n, n::VD.size]
        LOGGER.debug('pv jac at Vd = %g[V]:\n%r', VD[n], pv_jac_n)
        LOGGER.debug('pv jac AlgoPy at Vd = %g[V]:\n%r', VD[n], pv_jac_algopy_n)
        reldiff = old_div(pv_jac_n, pv_jac_algopy_n) - 1.0
        LOGGER.debug('reldiff at Vd = %g[V]:\n%r', VD[n], reldiff)
        resnorm = np.linalg.norm(reldiff)
        LOGGER.debug('resnorm at Vd = %g[V]: %r', VD[n], resnorm)
        rms = np.sqrt(np.sum(reldiff ** 2.0) / 9.0/ 3.0)
        LOGGER.debug('rms at Vd = %g[V]: %r', VD[n], rms)
        ok_(np.allclose(pv_jac_n, pv_jac_algopy_n, rtol=1e-3, atol=1e-3))
    return pv, pv_cov, pv_jac, pv_jac_algopy


def plot_pv(pv, pv_cov):
    """
    IV and PV 2-axis plot with errorbars 
    """
    i_pv, v_pv, p_pv = pv
    i_stdev = np.sqrt(pv_cov.diagonal()[::3])
    v_stdev = np.sqrt(pv_cov.diagonal()[1::3])
    p_stdev = np.sqrt(pv_cov.diagonal()[2::3]) 
    fig, ax1 = plt.subplots()
    ax1.errorbar(v_pv, i_pv, i_stdev, v_stdev)
    ax1.grid()
    ax1.set_xlabel('voltage [V]')
    ax1.set_ylabel('current [A]', color='b')
    ax1.set_ylim([0, 6.0])
    ax2 = ax1.twinx()
    ax2.errorbar(v_pv, p_pv, p_stdev, v_stdev, fmt='r')
    ax2.grid()
    ax2.set_ylabel('power [W]', color='r')
    ax2.set_ylim([0, 3.0])
    ax1.set_title('IV and PV curves')
    return fig


def plot_pv_jac(pv_jac, pv_jac_algopy, Vd=VD):
    """
    Log plot of relative difference between AlgoPy and central finite difference
    approximations

    :param pv_jac: central finite approximations
    :param pv_jac_algopy: automatic differentiation
    :param Vd: voltages
    :return: fig
    """
    fn = ['Cell Current, Ic [A]', 'Cell Voltage, Vc [V]', 'Cell Power, Pc [W]']
    fig, ax = plt.subplots(3, 1, **{'figsize': (8.0, 18.0)})
    colorcycle = [
        'firebrick', 'goldenrod', 'sage', 'lime', 'seagreen', 'turquoise',
        'royalblue', 'indigo', 'fuchsia'
    ]
    for m in range(3):
        for n in range(9):
            pv_jac_n = pv_jac[m::3, n::9].diagonal()
            pv_jac_algopy_n = pv_jac_algopy[
                m, :, n * 126:(n + 1) * 126
            ].diagonal()
            reldiff = np.abs(old_div(pv_jac_n, pv_jac_algopy_n) - 1.0)
            ax[m].semilogy(Vd, reldiff, colorcycle[n])
        ax[m].grid()
        ax[m].legend(
            ['Ee', 'Tc', 'Rs', 'Rsh', 'Isat1_0', 'Isat2', 'Isc0', 'alpha_Isc',
             'Eg'], fancybox=True, framealpha=0.5
        )
        ax[m].set_xlabel('Diode Voltage, Vd [V]')
        ax[m].set_ylabel('Relative Difference')
        ax[m].set_title(fn[m])
    plt.tight_layout()
    return fig


@unc_wrapper_args(1, 2, 3, 4, 5)
# indices specify positions of independent variables:
# 1: latitude, 2: longitude, 3: pressure, 4: altitude, 5: temperature
def spa(times, latitude, longitude, pressure, altitude, temperature):
    """
    Calculate solar position using PVLIB Cython wrapper around NREL SPA.

    :param times: list of times, must be localized as UTC
    :type times: :class:`pandas.DatetimeIndex`
    :param latitude: latitude [deg]
    :param latitude: longitude [deg]
    :param pressure: pressure [Pa]
    :param latitude: altitude [m]
    :param temperature: temperature [degC]
    :returns: zenith, azimuth
    """
    dataframe = pvlib.solarposition.spa_c(times, latitude, longitude, pressure,
                                          temperature)
    retvals = dataframe.to_records()
    zenith = retvals['apparent_zenith']
    zenith = np.where(zenith < 90, zenith, np.nan)
    azimuth = retvals['azimuth']
    return zenith, azimuth


def test_solpos(method='loop'):
    """
    Test solar position calculation using NREL's SOLPOS.
    """
    times = pd.DatetimeIndex(
        pd.date_range(start='2015/1/1', end='2015/1/2', freq='1h',
                      tz=PST)).tz_convert(UTC)
    latitude, longitude = 37.0, -122.0
    pressure, temperature = 101325.0, 22.0
    altitude = 0.0

    # standard deviation of 1% assuming normal distribution
    covariance = np.diag([0.0001] * 5)
    ze, az, cov, jac = spa(times, latitude, longitude, pressure, altitude,
                           temperature, __covariance__=covariance,
                           __method__=method)
    cov = jflatten(cov)
    jac = jflatten(jac)
    jac_nd = solpos_nd_jac(times, latitude, longitude, pressure, altitude,
                           temperature)
    for n in range(times.size):
        r, c = 2 * n, 5 * n
        # some rows which numdifftools returned nan
        if n in [0,  8, 17, 24]:
            continue
        ok_(np.allclose(jac[r:(r + 2), c:(c + 5)], jac_nd[:,:,n], equal_nan=True))
    return ze, az, cov, jac, jac_nd


if __name__ == '__main__':
    test_unc_wrapper()
    pv, pv_cov, pv_jac, pv_jac_algopy = test_IV()
    test_solpos()
    fig1 = plot_pv(pv, pv_cov)
    fig1.show()
    fig2 = plot_pv_jac(pv_jac, pv_jac_algopy)
    fig2.savefig('IV-PV-jac-errors.png')
    fig2.show()
