"""
Tests

SunPower Corp. (c) 2016
"""

from nose.tools import ok_
import numpy as np
from uncertainty_wrapper import unc_wrapper, unc_wrapper_args, logging
from scipy.constants import Boltzmann as KB, elementary_charge as QE
from datetime import datetime, timedelta
from solar_utils import solposAM
import pytz
import pint
from matplotlib import pyplot as plt


UREG = pint.UnitRegistry()
PST = pytz.timezone('US/Pacific')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
__all__ = ['ok_', 'np', 'unc_wrapper', 'unc_wrapper_args', 'KB', 'QE',
           'datetime', 'timedelta', 'solposAM', 'plt', 'UREG', 'PST', 'LOGGER']
