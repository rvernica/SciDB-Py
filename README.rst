SciDB-Py: Python Interface to SciDB
===================================
.. image:: https://travis-ci.org/Paradigm4/SciDB-Py.svg?branch=devel
    :target: https://travis-ci.org/Paradigm4/SciDB-Py

Requirements
------------

Python ``2.7.x``, ``3.4.x``, ``3.5.x``, ``3.6.x`` or newer.

Required Python packages::

  backports.weakref
  enum34
  numpy
  pandas
  requests
  six


Installation
------------

::

  pip install git+http://github.com/paradigm4/scidb-py.git@devel

Or to install in user space::

  > pip install --user git+http://github.com/paradigm4/scidb-py.git@devel
  Collecting git+http://github.com/paradigm4/scidb-py.git@devel
  Cloning http://github.com/paradigm4/scidb-py.git (to devel) to /tmp/pip-grcmmf-build
  Requirement already satisfied: backports.weakref in ./.local/lib/python2.7/site-packages (from scidb-py==16.9)
  Requirement already satisfied: enum34 in ./.local/lib/python2.7/site-packages (from scidb-py==16.9)
  Requirement already satisfied: numpy in ./.local/lib/python2.7/site-packages (from scidb-py==16.9)
  Requirement already satisfied: pandas in ./.local/lib/python2.7/site-packages (from scidb-py==16.9)
  Requirement already satisfied: requests in ./.local/lib/python2.7/site-packages (from scidb-py==16.9)
  Requirement already satisfied: six in /usr/lib/python2.7/site-packages (from scidb-py==16.9)
  Installing collected packages: scidb-py
    Running setup.py install for scidb-py ... done
  Successfully installed scidb-py-16.9


Documentation
-------------

See `SciDB-Py Documentation <http://paradigm4.github.io/SciDB-Py/>`_.
