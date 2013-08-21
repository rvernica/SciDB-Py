SciDB-py
========
- Author: Jake Vanderplas <jakevdp@cs.washington.edu>
- License: Simplified BSD

Python wrapper for SciDB queries

This package is still in active development.  Several pieces are still
incomplete, most notably the documentation.  This will be remedied in the
near-term!

Requirements
------------
SciDB-Py requires a working [SciDB]() installation, as well as a
[Shim]() network interface connected to the instance.  It requires
Python 2.6-2.7, and depends on 

Installation
------------
To install the latest release, use
```
pip install scidb-py
```

to install from source, type
```
python setup.py install
```

Depending on how your Python installation is set up, you
may need root priviledges for this.

Support
-------
This work has been supported by NSF Grant number 1226371_.


.. _1226371: http://www.nsf.gov/awardsearch/showAward?AWD_ID=1226371
