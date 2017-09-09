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
          'backports.weakref',
          'enum34',
          'numpy',
          'pandas',
          'requests',
          'six',
      ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Database :: Front-Ends',
          'Topic :: Scientific/Engineering',
      ],
)
