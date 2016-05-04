#! /usr/env/python

"""
get spa c source from NREL for travis
"""

import requests
import os
import logging

# logger
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__file__)
# constants
SPAC_URL = r'https://midcdmz.nrel.gov/apps/download.pl'  # spa.c source url
# register with NREL to download spa.c source
PAYLOAD = {
    'name': 'uncertainty wrapper',
    'country': 'US',
    'company': 'SunPower',
    'software': 'SPA'
}
SPAH_URL = r'http://midcdmz.nrel.gov/spa/spa.h'  # spa.h source url
PVLIB_PATH = os.environ['PVLIB_PATH']  # path to PVLIB on travis
SPA_C_FILES = os.path.join(PVLIB_PATH, 'spa_c_files')

if __name__ == "__main__":
    # get spa.c source
    LOGGER.debug('post payload to url: %s\n\tpayload:\n%s', SPAC_URL, PAYLOAD)
    r = requests.post(SPAC_URL, data=PAYLOAD)
    LOGGER.debug('post response: %r', r)
    # save spa.c source to PVLIB/SPA_C_FILES folder
    with open(os.path.join(SPA_C_FILES, 'spa.c'), 'wb') as f:
        f.write(r.content)
    LOGGER.debug('saved file: %r', f.name)
    # get spa.c source
    LOGGER.debug('get url: %s', SPAH_URL)
    r = requests.get(SPAH_URL)
    LOGGER.debug('get response: %r', r)
    # save spa.c source to PVLIB/SPA_C_FILES folder
    with open(os.path.join(SPA_C_FILES, 'spa.h'), 'wb') as f:
        f.write(r.content)
    LOGGER.debug('saved file: %r', f.name)
    LOGGER.debug('exiting %s', __file__)  # exit
