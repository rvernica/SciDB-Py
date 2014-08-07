from __future__ import print_function

"""
Code to serialize SciDB binary data streams into numpy arrays.
"""

# License: Simplified BSD, 2014
# See LICENSE.txt for more information

"""
Note:

This code will eventually relace the other parsing code in SciDB-py,
which is older and had limited support for missing data and
datetime datatypes. Once this module is sufficiently matured,
the old code will be deleted
"""

from collections import defaultdict
from itertools import groupby, cycle

import numpy as np

from .utils import _new_attribute_label

# byte format for binary scidb data
typemap = {'bool': np.dtype('<b1'),
           'int8': np.dtype('<b'),
           'uint8': np.dtype('<B'),
           'int16': np.dtype('<h'),
           'uint16': np.dtype('<H'),
           'int32': np.dtype('<i'),
           'uint32': np.dtype('<I'),
           'int64': np.dtype('<l'),
           'uint64': np.dtype('<L'),
           'float': np.dtype('<f4'),
           'double': np.dtype('<d'),
           'char': np.dtype('c'),
           'datetime': np.dtype('<M8[s]'),
           'datetimetz': np.dtype([('time', '<M8[s]'), ('tz', '<m8[s]')]),
           'string': np.dtype('object')
           }

null_typemap = dict(((k, False), v) for k, v in typemap.items())
null_typemap.update(((k, True), [('mask', '<b1'), ('data', v)])
                    for k, v in typemap.items())

# NULL value for each datatype
NULLS = defaultdict(lambda: np.nan)
NULLS['bool'] = np.float32(np.nan)
NULLS['string'] = None
NULLS['float'] = np.float32(np.nan)
NULLS['datetime'] = np.datetime64('NaT')
NULLS['datetimetz'] = np.datetime64('NaT')
NULLS['char'] = '\0'

for k in typemap:
    NULLS[typemap[k]] = NULLS[k]

# numpy datatype that each sdb datatype should be promoted to
# if nullable
NULL_PROMOTION = defaultdict(lambda: np.float)
NULL_PROMOTION['float'] = np.dtype('float32')
NULL_PROMOTION['datetime'] = np.dtype('<M8[s]')
NULL_PROMOTION['datetimetz'] = np.dtype('<M8[s]')
NULL_PROMOTION['char'] = np.dtype('c')
NULL_PROMOTION['string'] = object

# numpy datatype that each sdb datatype should be converted to
# if not nullable
mapping = typemap.copy()
mapping['datetimetz'] = np.dtype('<M8[s]')
mapping['string'] = object


def _fmt(array):
    return array.sdbtype.bytes_fmt


def _iter_strings(contents, nullable):
    """
    Iterate over the strings in an all-string sciDB array

    Parameters
    ----------
    contents : str
       The binary output of an all-string array

    nullable : list of booleans
       Whether each attribute in the array is nullable

    Yields
    ------
    A sequence of strings or None (for masked entries)
    Iterates over attributes in a cell, then over cells

    Notes
    -----
    All attributes in the input array *must* be strings. This isn't checked.
    """
    offset = 0

    nulls = cycle(nullable)
    while offset < len(contents):
        if next(nulls):
            masked = contents[offset] != b'\xff'[0]
            offset += 1
        else:
            masked = False

        sz = np.fromstring(contents[offset: offset + 4], '<i')
        offset += 4
        yield None if masked else contents[offset: offset + sz - 1].decode('utf-8')
        offset += sz  # skip null terminated string


def _string_attribute_dict(array):
    """
    Convert an all-string SciDB array into an attribute dict of numpy arrays

    Parameters
    -----------
    array : SciDBArray
        An array with 1 or more string attributes

    Returns
    -------
    dict : att name -> numpy array
    """
    contents = array.interface._scan_array(array.name, fmt=_fmt(array))
    nullable = [nullable for nm, typ, nullable in array.sdbtype.full_rep]

    result = np.array(list(_iter_strings(contents, nullable)), dtype=object)

    natt = len(array.att_names)
    return dict((att, result[i::natt])
                for i, att in enumerate(array.att_names))


def _to_attribute_dict(array):
    """
    Convert a non-string SciDB array into an attribute dict of numpy arrays

    Parameters
    -----------
    array : SciDBArray
       An array with 1 or more non-string attributes

    Returns
    -------
    dict : att name -> numpy array
    """

    contents = array.interface._scan_array(array.name, fmt=_fmt(array))
    dtype = [(nm, null_typemap[t, nullable])
             for nm, t, nullable in array.sdbtype.full_rep]
    data = np.fromstring(contents, dtype=dtype)

    # process nullable attributes
    result = {}
    for nm, typ, nullable in array.sdbtype.full_rep:
        att = data[nm]

        if not nullable:
            result[nm] = att
        else:
            good = att['mask'] == 255
            result[nm] = np.where(good, att['data'], NULLS[typ])

        if typ == 'datetimetz':
            result[nm] = result[nm]['time'] - result[nm]['tz']

    return result


def toarray(array):
    """
    Convert a SciDBArray to a numpy array

    Notes:
    index 0 of output array is aligned with the lower bound
    of the scidb array.
    """
    dtype = [(nm, mapping[d] if not n else NULL_PROMOTION[d])
             for (nm, d, n) in array.datashape.sdbtype.full_rep]

    ind = _new_attribute_label('ind', array)
    unpacked = array.unpack(ind).eval()

    # partition string and non-string attributes. Download and process
    # separately for speed
    atts = {}
    for isstring, dt in groupby(unpacked.sdbtype.full_rep, lambda r: r[1] == 'string'):
        subarray = unpacked.project(*[nm for (nm, _, _) in dt])
        if isstring:
            a = _string_attribute_dict(subarray)
        else:
            a = _to_attribute_dict(subarray)
        atts.update(**a)

    # shift nonzero origins
    inds = tuple(atts[d] - lo for
                 d, lo in zip(array.dim_names,
                              array.datashape.dim_low))

    # determine shape and dtype of final result
    shp = array.shape
    if shp is None:  # unbound array
        shp = tuple([i.max() + 1 for i in inds])
    dtype = [(nm, atts[nm].dtype)
             for (nm, d, n) in array.datashape.sdbtype.full_rep]
    result = np.zeros(shp, dtype)

    # populate the array
    for att in result.dtype.names:
        result[att][inds] = atts[att]

    # For convenience:
    # cast single-attribute record arrays into plain numpy arrays
    if len(result.dtype) == 1:
        result = result[result.dtype.names[0]]

    return result
