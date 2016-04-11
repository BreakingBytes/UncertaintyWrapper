"""
Tests

SunPower Corp. (c) 2016
"""

from nose.tools import ok_, eq_, raises
import numpy as np
from uncertainty_wrapper import unc_wrapper
import logging
from scipy.constants import Boltzmann as KB, elementary_charge as QE

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def test_unc_wrapper():
    """
    Test uncertainty wrapper
    """
    x, cov = np.array([[1.0]]), np.array([[0.1]])
    
    @unc_wrapper
    def f(y):
        return np.exp(y)
    
    avg, var, jac = f(x, __covariance__=cov)
    LOGGER.debug("average = %g", avg)
    LOGGER.debug("variance = %g", var)
    ok_(np.isclose(avg, np.exp(x)))
    ok_(np.isclose(var, cov * np.exp(x) ** 2))
    ok_(np.isclose(jac, np.exp(x)))
    return avg, var, jac


def IV(x, Vd, E0=1000, T0=298.15, kB=KB, qe=QE):
    Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg = x
    Vt = Tc * kB / qe
    Isc = Ee * Isc0 * (1.0 + (Tc - T0) * alpha_Isc)
    Isat1 = (
        Isat1_0 * (Tc ** 3.0 / T0 ** 3.0) *
        np.exp(Eg * qe / kB * (1.0 / T0 - 1.0 / Tc))
    )
    Vd_sc = Isc * Rs  # at short circuit Vc = 0 
    Id1_sc = Isat1 * (np.exp(Vd_sc / Vt) - 1.0)
    Id2_sc = Isat2 * (np.exp(Vd_sc / 2.0 / Vt) - 1.0)
    Ish_sc = Vd_sc / Rsh
    Iph = Isc + Id1_sc + Id2_sc + Ish_sc
    Id1 = Isat1 * (np.exp(Vd / Vt) - 1.0)
    Id2 = Isat2 * (np.exp(Vd / 2.0 / Vt) - 1.0)
    Ish = Vd / Rsh
    Ic = Iph - Id1 - Id2 - Ish
    Vc = Vd - Ic * Rs
    return np.array([Ic, Vc, Ic * Vc])


def Voc(x, E0=1000, T0=298.15, kB=KB, qe=QE):
    Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg = x
    msg = ['Ee=%g[suns]','Tc=%g[K]','Rs=%g[ohms]','Rsh=%g[ohms]',
           'Isat1_0=%g[A]','Isat2=%g[A]','Isc0=%g[A]','alpha_Isc=%g[]',
           'Eg=%g[eV]']
    LOGGER.debug('\n' + '\n'.join(msg) + '\n', *x)
    Vt = Tc * kB / qe
    LOGGER.debug('Vt=%g[V]', Vt)
    Isc = Ee * Isc0 * (1.0 + (Tc - T0) * alpha_Isc)
    LOGGER.debug('Isc=%g[A]', Isc)
    Isat1 = (
        Isat1_0 * (Tc ** 3.0 / T0 ** 3.0) *
        np.exp(Eg * qe / kB * (1.0 / T0 - 1.0 / Tc))
    )
    LOGGER.debug('Isat1=%g[A]', Isat1)
    Vd_sc = Isc * Rs  # at short circuit Vc = 0 
    Id1_sc = Isat1 * (np.exp(Vd_sc / Vt) - 1.0)
    Id2_sc = Isat2 * (np.exp(Vd_sc / 2.0 / Vt) - 1.0)
    Ish_sc = Vd_sc / Rsh
    Iph = Isc + Id1_sc + Id2_sc + Ish_sc
    # estimate Voc
    delta = Isat2 ** 2.0 + 4.0 * Isat1 * (Iph + Isat1 + Isat2)
    return Vt * np.log(((-Isat2 + np.sqrt(delta)) / 2.0 / Isat1) ** 2.0)


RS = 0.004267236774264931  # [ohm] series resistance
RSH = 10.01226369025448  # [ohm] shunt resistance
ISAT1_0 = 2.286188161253440E-11  # [A] diode one saturation current
ISAT2 = 1.117455042372326E-6  # [A] diode two saturation current
ISC0 = 6.3056  # [A] reference short circuit current
EE = 0.8
TC = 323.15
EG = 1.1
ALPHA_ISC = 0.0003551
VOC = Voc((EE, TC, RS, RSH, ISAT1_0, ISAT2, ISC0, ALPHA_ISC, EG))
assert np.isclose(VOC, 0.62816490891656673)
LOGGER.debug('Voc = %g[V]', VOC)
VD = np.arange(0, VOC, 0.005)
X = np.array([EE, TC, RS, RSH, ISAT1_0, ISAT2, ISC0, ALPHA_ISC, EG])
COV = np.diag(np.random.rand(X.size) * X / 10.0)
X = X.reshape(-1, 1).repeat(VD.size, axis=1)

def test_IV():
    f = unc_wrapper(IV)
    return f(X, COV, VD)


if __name__ == '__main__':
    test_unc_wrapper()
