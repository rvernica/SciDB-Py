try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command


class PyTest(Command):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        # Command.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        # Command.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        import sys
        errno = pytest.main(self.pytest_args + ['scidbpy'])
        sys.exit(errno)


DESCRIPTION = "Python wrappers for SciDB"
LONG_DESCRIPTION = open('README.rst').read()
NAME = "scidb-py"
AUTHOR = "Jake Vanderplas, Chris Beaumont, Kriti Sen Sharma"
AUTHOR_EMAIL = "jakevdp@cs.washington.edu, cbeaumont@cfa.harvard.edu, ksen@paradigm4.com"
MAINTAINER = "Kriti Sen Sharma"
MAINTAINER_EMAIL = "cbeaumont@cfa.harvard.edu"
DOWNLOAD_URL = 'http://github.com/paradigm4/scidb-py'
LICENSE = 'Simplified BSD'

import scidbpy
VERSION = scidbpy.__version__

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['scidbpy', 'scidbpy.tests'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Topic :: Database :: Front-Ends',
          'Topic :: Scientific/Engineering'],
      cmdclass={'test': PyTest}
      )
