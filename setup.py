#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import scidbpy


NAME = 'scidb-py'
DESCRIPTION = 'Python library for SciDB'
AUTHOR = 'Rares Vernica'
AUTHOR_EMAIL = 'rvernica@gmail.com'
DOWNLOAD_URL = 'http://github.com/Paradigm4/SciDB-Py'
LICENSE = 'Simplified BSD'
VERSION = scidbpy.__version__

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['scidbpy'],
      install_requires=[
          'enum34',
          'numpy',
          'pandas',
          'requests',
          'six',
      ],
      )
