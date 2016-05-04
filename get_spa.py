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

if __name__ == "__main__":
    # get spa.c source
    LOGGER.debug('post payload to url: %s\n\tpayload:\n%s', SPAC_URL, PAYLOAD)
    r = requests.post(SPAC_URL, data=PAYLOAD)
    LOGGER.debug('post response: %r', r)
    # save spa.c source to PVLIB spa_c_files folder
    with open(os.path.join(PVLIB_PATH, 'spa.c'), 'wb') as f:
        f.write(r.content)
    LOGGER.debug('saved file: %r to PVLIB PATH: %s', f.name, PVLIB_PATH)
    # get spa.c source
    LOGGER.debug('get url: %s', SPAH_URL)
    r = requests.get(SPAH_URL)
    LOGGER.debug('get response: %r', r)
    # save spa.c source to PVLIB spa_c_files folder
    with open(os.path.join(PVLIB_PATH, 'spa.h'), 'wb') as f:
        f.write(r.content)
    LOGGER.debug('saved file: %r to PVLIB PATH: %s', f.name, PVLIB_PATH)
    LOGGER.debug('exiting %s', __file__)  # exit
