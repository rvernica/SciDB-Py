# License: Simplified BSD, 2014
# See LICENSE.txt for more information

"""
This is a lightweight utility module which streamlines Python 2/Python 3
compatibility.  It uses a few routines from the six.py package, as well
as some other custom routines.
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import sys
import csv

import numpy as np
PYTHON3 = sys.version_info[0] == 3

__all__ = ["PYTHON3", "string_type", "iteritems", "genfromstr",
           "csv_reader", "stringio", "dtype"]

if PYTHON3:
    string_type = str
    _iteritems = "items"
    from urllib.request import urlopen
    from urllib.error import HTTPError
    from urllib.parse import quote
    from functools import reduce
    dtype = np.dtype
else:
    string_type = basestring
    _iteritems = "iteritems"
    from urllib2 import urlopen, quote, HTTPError
    reduce = reduce

    def dtype(desc):
        if not isinstance(desc, list):
            return np.dtype(desc)
        desc = [(str(n), t) for n, t in desc]
        return np.dtype(desc)


def iteritems(D, **kwargs):
    """Return an iterator over the (key, value) pairs of a dictionary."""
    return iter(getattr(D, _iteritems)(**kwargs))


def stringio(s):
    if PYTHON3:
        from io import BytesIO
        return BytesIO(s.encode())
    from cStringIO import StringIO
    return StringIO(s)


def genfromstr(s, **kwargs):
    """Utility routine to create an array from a string.

    This uses either StringIO or BytesIO, depending on whether we're on
    Python 2 or Python 3.
    """
    return np.genfromtxt(stringio(s), **kwargs)


def csv_reader(txt, skiplines=0, **kwargs):
    """
    Wrapper around the built-in csv.reader, to handle unicode issues.

    Parameters
    ----------
    txt : string
       The content of the CSV data to parse
    skiplines : int, optional
       The number of initial lines to skip
    **kwargs: dict
       Extra keywords are passed to csv.reader

    Returns
    -------
    An iterator over the parsed rows, each a list of unicode strings.
    """

    buff = stringio(txt)
    for _ in range(skiplines):
        next(buff)

    if PYTHON3:
        buff = (t.decode('utf8') for t in buff)
        for line in csv.reader(buff, **kwargs):
            yield line
    else:
        for line in csv.reader(buff, **kwargs):
            yield map(lambda x: x.decode('utf8'), line)
