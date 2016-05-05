"""
Tests

SunPower Corp. (c) 2016
"""

from nose.tools import ok_
import numpy as np
from uncertainty_wrapper.core import (
    unc_wrapper, unc_wrapper_args, jflatten, logging
)
from scipy.constants import Boltzmann as KB, elementary_charge as QE
import pytz
import pint
from matplotlib import pyplot as plt
import pandas as pd
import pvlib


UREG = pint.UnitRegistry()
PST = pytz.timezone('US/Pacific')
UTC = pytz.UTC
T0 = 298.15  # [K] reference temperature
__all__ = ['ok_', 'np', 'pd', 'unc_wrapper', 'unc_wrapper_args', 'jflatten',
           'logging', 'KB', 'QE', 'plt', 'pvlib', 'UREG', 'PST', 'UTC', 'T0']
