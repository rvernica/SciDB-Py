SciDB-py: Python Interface to SciDB
===================================
SciDB-py is a full-featuered Python interface to SciDB.  Its goal is to
allow users to utilize the power of SciDB_ from Python, using a familiar
and intuitive numpy-like syntax.  For more information, see the
SciDB Documentation_.

- Author: Jake Vanderplas <jakevdp@cs.washington.edu>
- License: Simplified BSD
- Documentation: http://jakevdp.github.io/SciDB-py/

Requirements
------------
SciDB-Py requires a working SciDB_ installation, as well as a
Shim_ network interface connected to the instance.  It requires
Python 2.6-2.7 or 3.3.

Package Dependencies
--------------------
SciDB-Py has several Python package dependencies:

NumPy_
    tested with version 1.6-1.7.

Requests_
    tested with version 1.2.
    Required for using the Shim interface to SciDB.

Pandas_ (optional)
    tested with version 0.10.
    Required only for importing/exporting SciDB arrays
    as Pandas Dataframe objects.

SciPy_ (optional)
    tested with versions 0.10-0.12.
    Required only for importing/exporting SciDB arrays
    as SciPy sparse matrices.

Installation
------------
For full installation information, please see the Documentation_.

To install the latest stable release via the Python Package Index, use
```
pip install scidb-py
```

to install from source, download the Source_ and type
```
python setup.py install
```

Depending on how your Python installation is set up, you
may need root privileges for this.

Support
-------
This work has been supported by NSF Grant number 1226371_ and by
Paradigm4_.


.. _1226371: http://www.nsf.gov/awardsearch/showAward?AWD_ID=1226371
.. _Paradigm4: http://www.paradigm4.com
.. _NumPy: http://www.numpy.org
.. _Requests: http://www.python-requests.org/en/latest/
.. _SciPy: http://www.scipy.org
.. _Pandas: http://pandas.pydata.org/
.. _Shim: http://github.com/paradigm4/shim
.. _SciDB: http://scidb.org/
.. _Documentation: http://jakevdp.github.io/SciDB-py/
.. _Source: http://github.com/jakevdp/SciDB-py
