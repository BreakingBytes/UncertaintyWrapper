#! /usr/env/python

import requests
import os

SPAC_URL = r'https://midcdmz.nrel.gov/apps/download.pl'
PAYLOAD = {
    'name': 'uncertainty wrapper',
    'country': 'US',
    'company': 'SunPower',
    'software': 'SPA'
}
SPAH_URL = r'http://midcdmz.nrel.gov/spa/spa.h'
PVLIB_PATH = os.environ['PVLIB_PATH']

if __name__ == "__main__":
    r = requests.post(SPAC_URL, data=PAYLOAD)
    with open(os.path.join(PVLIB_PATH, 'spa.c'), 'wb') as f:
        f.write(r.content); f.close()
    r = requests.get(SPAH_URL)
    with open(os.path.join(PVLIB_PATH, 'spa.h'), 'wb') as f:
        f.write(r.content); f.close()
