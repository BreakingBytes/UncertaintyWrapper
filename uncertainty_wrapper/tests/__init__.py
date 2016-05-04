"""
Tests

SunPower Corp. (c) 2016
"""

from nose.tools import ok_
import numpy as np
from uncertainty_wrapper import unc_wrapper, unc_wrapper_args, logging
from uncertainty_wrapper.core import jflatten
from scipy.constants import Boltzmann as KB, elementary_charge as QE
# from datetime import datetime, timedelta
import pytz
import pint
from matplotlib import pyplot as plt
import pandas as pd
import pvlib


UREG = pint.UnitRegistry()
PST = pytz.timezone('US/Pacific')
UTC = pytz.UTC
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
__all__ = ['ok_', 'np', 'pd', 'unc_wrapper', 'unc_wrapper_args', 'KB', 'QE',
           'pvlib', 'plt', 'UREG', 'PST', 'UTC', 'jflatten', 'LOGGER']
