"""
Test AlgoPy and numdifftools
"""
from __future__ import division

from past.utils import old_div
from algopy import UTPM, exp, log, sqrt, zeros
import numdifftools as nd
from uncertainty_wrapper.tests import KB, QE, np, pvlib, T0


def IV_algopy(x, Vd):
    """
    IV curve implemented using algopy instead of numpy
    """
    nobs = x.shape[1]
    out = zeros((3, nobs), dtype=x)
    Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg = x
    Vt = old_div(Tc * KB, QE)
    Isc = Ee * Isc0 * (1.0 + (Tc - T0) * alpha_Isc)
    Isat1 = (
        Isat1_0 * (old_div(Tc ** 3.0, T0 ** 3.0)) *
        exp(old_div(Eg * QE, KB) * (1.0 / T0 - 1.0 / Tc))
    )
    Vd_sc = Isc * Rs  # at short circuit Vc = 0 
    Id1_sc = Isat1 * (exp(old_div(Vd_sc, Vt)) - 1.0)
    Id2_sc = Isat2 * (exp(Vd_sc / 2.0 / Vt) - 1.0)
    Ish_sc = old_div(Vd_sc, Rsh)
    Iph = Isc + Id1_sc + Id2_sc + Ish_sc
    Id1 = Isat1 * (exp(old_div(Vd, Vt)) - 1.0)
    Id2 = Isat2 * (exp(Vd / 2.0 / Vt) - 1.0)
    Ish = old_div(Vd, Rsh)
    Ic = Iph - Id1 - Id2 - Ish
    Vc = Vd - Ic * Rs
    out[0] = Ic
    out[1] = Vc
    out[2] = Ic * Vc
    return out


def IV_algopy_jac (Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg, Vd):
    """
    Calculate Jacobian of IV curve using AlgoPy

    :param Ee: [suns] effective irradiance
    :param Tc: [C] cell temperature
    :param Rs: [ohms] series resistance
    :param Rsh: [ohms] shunt resistance
    :param Isat1_0: [A] saturation current of first diode at STC
    :param Isat2: [A] saturation current of second diode
    :param Isc0: [A] short circuit current at STC
    :param alpha_Isc: [1/K] short circuit current temperature coefficient
    :param Eg: [eV] band gap
    :param Vd: [V] diode voltages
    :return: Jacobian :math:`\\frac{\\partial f_i}{\\partial x_{j,k}}`
        where :math:`k` are independent observations of :math:`x`
    """
    x = UTPM.init_jacobian([
        Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg
    ])
    return UTPM.extract_jacobian(IV_algopy(x, Vd))



def spa(times, latitude, longitude, pressure, altitude, temperature):
    dataframe = pvlib.solarposition.spa_c(times, latitude, longitude, pressure,
                                          temperature)
    retvals = dataframe.to_records()
    zenith = retvals['apparent_zenith']
    zenith = np.where(zenith < 90, zenith, np.nan)
    azimuth = retvals['azimuth']
    return zenith, azimuth


def solpos_nd_jac(times, latitude, longitude, pressure, altitude, temperature):
    """

    :param times: timestamps
    :param latitude: [deg] latitude
    :param longitude: [deg] longitude
    :param pressure: [mbar] pressure
    :param altitude: [m] elevation above sea level
    :param temperature: [C] ambient temperature
    :return: Jacobian estimated using ``numdifftools``
    """

    def f(x, times):
        latitude, longitude, pressure, altitude, temperature = x
        return np.array(spa(times, latitude, longitude, pressure, altitude,
                            temperature))

    j = nd.Jacobian(f)
    x = np.array([latitude, longitude, pressure, altitude, temperature])
    return j(x, times).squeeze()
