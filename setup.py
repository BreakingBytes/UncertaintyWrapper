"""
To install UncertaintyWrapper from source, a cloned repository or an archive,
use ``python setup.py install``.

Use ``python setup.py bdist_wheel`` to make distribute as a wheel.bdist_wheel.
"""

# try to use setuptools if available, otherwise fall back on distutils
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from uncertainty_wrapper import (
    __VERSION__, __AUTHOR__, __EMAIL__, __URL__
)

import os

README = 'README.rst'
try:
    with open(os.path.join(os.path.dirname(__file__), README), 'r') as readme:
        README = readme.read()
except IOError:
    pass

setup(name='UncertaintyWrapper',
      version=__VERSION__,
      description='Uncertainty wrapper using estimated Jacobian',
      long_description=README,
      author=__AUTHOR__,
      author_email=__EMAIL__,
      url=__URL__,
      packages=['uncertainty_wrapper', 'uncertainty_wrapper.tests'],
      requires=['numpy (>=1.8)', 'nose', 'sphinx'],
      package_data={'uncertainty_wrapper': [
          'docs/conf.py', 'docs/*.rst', 'docs/Makefile', 'docs/make.bat'
      ]})
