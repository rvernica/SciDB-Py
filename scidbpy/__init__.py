"""
Usage
-----

>>> import scidbpy

>>> db = scidbpy.connect()

>>> print(db)
scidb_url  = 'http://localhost:8080'
scidb_auth = None
http_auth  = None
role       = None
namespace  = None
verify     = None
"""

from .db import connect, iquery

__version__ = '16.9'
