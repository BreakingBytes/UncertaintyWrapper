"""
Test AlgoPy and numdifftools
"""

from algopy import UTPM, exp, log, sqrt, zeros
import numdifftools as nd
from uncertainty_wrapper.tests import KB, QE, solposAM, np, timedelta


def IV_algopy(x, Vd, E0=1000, T0=298.15, kB=KB, qe=QE):
    """
    IV curve implemented using algopy instead of numpy
    """
    nobs = x.shape[1]
    out = zeros((3, nobs), dtype=x)
    Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg = x
    Vt = Tc * kB / qe
    Isc = Ee * Isc0 * (1.0 + (Tc - T0) * alpha_Isc)
    Isat1 = (
        Isat1_0 * (Tc ** 3.0 / T0 ** 3.0) *
        exp(Eg * qe / kB * (1.0 / T0 - 1.0 / Tc))
    )
    Vd_sc = Isc * Rs  # at short circuit Vc = 0 
    Id1_sc = Isat1 * (exp(Vd_sc / Vt) - 1.0)
    Id2_sc = Isat2 * (exp(Vd_sc / 2.0 / Vt) - 1.0)
    Ish_sc = Vd_sc / Rsh
    Iph = Isc + Id1_sc + Id2_sc + Ish_sc
    Id1 = Isat1 * (exp(Vd / Vt) - 1.0)
    Id2 = Isat2 * (exp(Vd / 2.0 / Vt) - 1.0)
    Ish = Vd / Rsh
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


def solar_position(lat, lon, press, tamb, timestamps, seconds=0):
    """
    calculate solar position
    """
    seconds = np.sign(seconds) * np.ceil(np.abs(seconds))
    # seconds = np.where(x > 0, np.ceil(seconds), np.floor(seconds))
    try:
        ntimestamps = len(timestamps)
    except TypeError:
        ntimestamps = 1
        timestamps = [timestamps]
    an, am = np.zeros((ntimestamps, 2)), np.zeros((ntimestamps, 2))
    for n, ts in enumerate(timestamps):
        utcoffset = ts.utcoffset()
        dst = ts.dst()
        if None in (utcoffset, dst):
            tz = 0.0  # assume UTC if naive
        else:
            tz = (utcoffset.total_seconds() - dst.total_seconds()) / 3600.0
        loc = [lat, lon, tz]
        dt = ts + timedelta(seconds=seconds.item())
        dt = dt.timetuple()[:6]
        an[n], am[n] = solposAM(loc, dt, [press, tamb])
    return an[:, 0], an[:, 1], am[:, 0], am[:, 1]


def solpos_nd_jac(lat, lon, press, tamb, dt, seconds):
    """

    :param lat: [deg] latitude
    :param lon: [deg] longitude
    :param press: [mbar] pressure
    :param tamb: [C] ambient temperature
    :param dt: datetime
    :param seconds: [s] seconds
    :type seconds: int
    :return: Jacobian estimated using ``numdifftools``
    """

    def f(x, dt):
        lat, lon, press, tamb, seconds = x
        return np.array(solar_position(lat, lon, press, tamb, dt, seconds))

    j = nd.Jacobian(f)
    x = np.array([q.magnitude for q in (lat, lon, press, tamb, seconds)])
    return j(x, dt).squeeze()
