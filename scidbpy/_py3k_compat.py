"""
This is a lightweight utility module which streamlines Python 2/Python 3
compatibility.  It uses a few routines from the six.py package, as well
as some other custom routines.
"""
import sys
import numpy as np
PYTHON3 = sys.version_info[0] == 3

__all__ = ["PYTHON3", "string_type", "iteritems", "genfromstr"]

if PYTHON3:
    string_type = str
    _iteritems = "items"
    from urllib.request import urlopen
    from urllib.error import HTTPError
    from urllib.parse import quote
else:
    string_type = basestring
    _iteritems = "iteritems"
    from urllib2 import urlopen, quote, HTTPError

def iteritems(D, **kwargs):
    """Return an iterator over the (key, value) pairs of a dictionary."""
    return iter(getattr(D, _iteritems)(**kwargs))


def genfromstr(s, **kwargs):
    """Utility routine to create an array from a string.

    This uses either StringIO or BytesIO, depending on whether we're on
    Python 2 or Python 3.
    """
    if PYTHON3:
        from io import BytesIO
        return np.genfromtxt(BytesIO(s.encode()), **kwargs)
    else:
        from cStringIO import StringIO
        return np.genfromtxt(StringIO(s), **kwargs)
