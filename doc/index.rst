.. SciDB-Py documentation master file, created by
   sphinx-quickstart on Mon Jul  8 21:22:16 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SciDB-Py documentation
======================

.. warning::

    This documentation is for the **legacy** version of the SciDB-Py
    library. GitHub pull requests are still accepted for the previous
    versions, but the code is not actively maintained. The current
    version of the SciDB-Py library is **16.9.1**. This version has
    been released in `September 2017` and has been rewritten entirely
    from scratch. This version is **not compatible** with the previous
    versions of the library. The documentation for the current version
    is available at `SciDB-Py documentation
    <http://paradigm4.github.io/SciDB-Py/>`

SciDB-Py is a Python interface to the SciDB_, the massively scalable
array-oriented database.  SciDB features include ACID transactions, parallel processing, distributed storage, efficient sparse array storage, and native parallel linear algebra operations.

The SciDB-Py package provides an intuitive NumPy-like interface
to SciDB, so that users can leverage powerful distributed, data-parallel
scientific computing from the comfort of their Python interpreter::

    from scidbpy import connect
    sdb = connect()              # connect to the database
    x = sdb.random((1000, 1000)) # 2D array of random numbers
    y = (x ** 2 + 3).sum()       # NumPy syntax, computed in the database

Contents
========
.. toctree::
   :maxdepth: 2

   whats_new.rst
   install
   tutorial
   demos
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _SciDB: http://scidb.org/
