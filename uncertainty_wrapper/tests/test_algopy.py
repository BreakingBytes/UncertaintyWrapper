"""
Test AlgoPy and numdifftools
"""

from algopy import UTPM, exp, log, sqrt, zeros
import numdifftools as nd
from uncertainty_wrapper.tests import KB, QE


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
    x = UTPM.init_jacobian([
        Ee, Tc, Rs, Rsh, Isat1_0, Isat2, Isc0, alpha_Isc, Eg
    ])
    return UTPM.extract_jacobian(IV_algopy(x, Vd))
    


if __name__ == '__main__':
    pass
