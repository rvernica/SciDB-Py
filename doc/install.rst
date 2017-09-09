.. _installing_scidbpy:

===================
Installing SciDB-Py
===================


Software prerequisites
----------------------

The scidbpy package requires at least:

1. An available SciDB_ installation
2. The Shim_ network interface to SciDB

We assume an existing installation of SciDB is available. Binary SciDB packages
(for Ubuntu 12.04 and RHEL/CentOS 6) and source code are available from
http://scidb.org.  The examples in this tutorial assume that SciDB is running
on a computer with host name "localhost," at port 8080.
If SciDB is not running on localhost, adjust the name accordingly.

The scidbpy package requires installation of a simple HTTP network service
called "shim" on the computer that SciDB coordinator is installed on. The
network service only needs to be installed on the SciDB computer, not on client
computers that connect to SciDB from Python. It's available in packaged binary
form for supported SciDB operating systems, and as source code which can be
compiled and deployed on any SciDB installation.
See http://github.com/paradigm4/shim  for source code and installation
instructions.


Python Prerequisites
--------------------
SciDB-Py requires Python 2.6-2.7 or 3.3, as well as NumPy_ and Requests_.
Some (optional) functionality requires SciPy_ and Pandas_.
Following are a description of these requirements:

NumPy_
    tested with version 1.9.

Requests_
    tested with version 2.7.
    Required for using the Shim interface to SciDB.

Pandas_ (optional)
    tested with version 0.15.
    Required only for importing/exporting SciDB arrays
    as Pandas Dataframe objects.

SciPy_ (optional)
    tested with versions 0.10-0.12.
    Required only for importing/exporting SciDB arrays
    as SciPy sparse matrices.


SciDB-Py Package Installation
-----------------------------

The latest release of ``scidb-py`` can be installed from the Python package index::

   pip install scidb-py==16.9.dev0

Install the development package directly from Github with::

    pip install git+http://github.com/paradigm4/scidb-py.git@legacy

The legacy release of ``scidb-py`` can be installed from the github "http://github.com/paradigm4/scidb-py", "legacy" branch by downloading the code and typing::

    python setup.py install

.. _Shim: http://github.com/paradigm4/shim

.. _SciDB: http://scidb.org/

.. _NumPy: http://www.numpy.org

.. _SciPy: http://www.scipy.org

.. _Pandas: http://pandas.pydata.org/

.. _Requests: http://docs.python-requests.org/
