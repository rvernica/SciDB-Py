from distutils.core import setup

DESCRIPTION = "Python wrappers for SciDB"
LONG_DESCRIPTION = open('README.md').read()
NAME = "SciDB-Py"
AUTHOR = "Jake Vanderplas"
AUTHOR_EMAIL = "jakevdp@cs.washington.edu"
MAINTAINER = "Jake VanderPlas"
MAINTAINER_EMAIL = "jakevdp@cs.washington.edu"
DOWNLOAD_URL = 'http://github.com/jakevdp/scidb-py'
LICENSE = 'GPL'

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
      packages=['scidbpy'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.6',
        'Topic :: Database :: Front-Ends',
        'Topic :: Scientific/Engineering'],
     )
