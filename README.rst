SciDB-Py: Python Interface to SciDB
===================================
.. image:: https://travis-ci.org/Paradigm4/SciDB-Py.svg?branch=devel
    :target: https://travis-ci.org/Paradigm4/SciDB-Py

Install
-------

::

   pip install git+http://github.com/paradigm4/scidb-py.git@devel

Or to install in user space::

   > pip install --user git+http://github.com/paradigm4/scidb-py.git@devel
   Collecting git+http://github.com/paradigm4/scidb-py.git@devel
   Cloning http://github.com/paradigm4/scidb-py.git (to devel) to /tmp/pip-odPNuK-build
   Requirement already satisfied (use --upgrade to upgrade): enum34 in ./.local/lib/python2.7/site-packages (from scidb-py==16.9)
   Requirement already satisfied (use --upgrade to upgrade): numpy in ./.local/lib/python2.7/site-packages (from scidb-py==16.9)
   Requirement already satisfied (use --upgrade to upgrade): pandas in ./.local/lib/python2.7/site-packages (from scidb-py==16.9)
   Requirement already satisfied (use --upgrade to upgrade): requests in ./.local/lib/python2.7/site-packages (from scidb-py==16.9)
   Requirement already satisfied (use --upgrade to upgrade): python-dateutil in ./.local/lib/python2.7/site-packages (from pandas->scidb-py==16.9)
   Requirement already satisfied (use --upgrade to upgrade): pytz>=2011k in ./.local/lib/python2.7/site-packages (from pandas->scidb-py==16.9)
   Requirement already satisfied (use --upgrade to upgrade): six>=1.5 in ./.local/lib/python2.7/site-packages (from python-dateutil->pandas->scidb-py==16.9)
   Installing collected packages: scidb-py
     Running setup.py install for scidb-py ... done
   Successfully installed scidb-py-16.9


Documentation
-------------

See `SciDB-Py Documentation <http://paradigm4.github.io/SciDB-Py/>`_.
