SciDB-py
========
- Author: Jake Vanderplas <jakevdp@cs.washington.edu>
- License: Simplified BSD
- Documentation: http://jakevdp.github.io/SciDB-py/

**A Python wrapper for SciDB queries.**

This package is still in active development.  Several pieces are still
incomplete, most notably the documentation.  This will be remedied in the
near-term!

Requirements
------------
SciDB-Py requires a working [SciDB]() installation, as well as a
[Shim]() network interface connected to the instance.  It requires
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
To install the latest release, use
```
pip install scidb-py
```

to install from source, download the source and type
```
python setup.py install
```

Depending on how your Python installation is set up, you
may need root priviledges for this.

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
