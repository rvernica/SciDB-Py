from __future__ import absolute_import, print_function, division, unicode_literals

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
from itertools import groupby, cycle, product

import numpy as np
from .utils import as_list

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
           'datetimetz': np.dtype([(str('time'), '<M8[s]'), (str('tz'), '<m8[s]')]),
           'string': np.dtype('object')
           }

null_typemap = dict(((k, False), v) for k, v in typemap.items())
null_typemap.update(((k, True), [(str('mask'), np.dtype('<B')), (str('data'), v)])
                    for k, v in typemap.items())

# NULL value for each datatype
NULLS = defaultdict(lambda: np.nan)
NULLS['bool'] = np.float32(np.nan)
NULLS['string'] = None
NULLS['float'] = np.float32(np.nan)
NULLS['datetime'] = np.datetime64('NaT')
NULLS['datetimetz'] = np.zeros(1, dtype=typemap['datetimetz'])
NULLS['datetimetz']['time'] = np.datetime64('NaT')
NULLS['datetimetz']['tz'] = np.datetime64('NaT')
NULLS['datetimetz'] = NULLS['datetimetz'][0]
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


def _scidb_serialize(arr, chunk_size):
    """
    Serialize a multidimensional numpy array into a 1D array,
    with an interleaving scheme that matches SciDB.

    Such a scheme first interleaves chunks in C-contiguous order.
    Then it interleaves cells in the chunk in C-contiguous order.
    """
    chunk_size = as_list(chunk_size)
    if len(chunk_size) == 1:
        chunk_size = chunk_size * arr.ndim
    result = []
    for start in product(*(range(0, s, c)
                           for s, c in zip(arr.shape, chunk_size))):
        slices = tuple(slice(i, i + s)
                       for i, s in zip(start, chunk_size))
        result.append(arr[slices].ravel())
    return np.hstack(result)


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


def _string_attribute_dict(array, **kwargs):
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
    contents = array.interface._scan_array(array.name, fmt=_fmt(array), **kwargs)
    nullable = [nullable for nm, typ, nullable in array.sdbtype.full_rep]

    result = np.array(list(_iter_strings(contents, nullable)), dtype=object)

    natt = len(array.att_names)
    return dict((att, result[i::natt])
                for i, att in enumerate(array.att_names))


def _nonstring_attribute_dict(array, **kwargs):
    """
    Convert a non-string SciDB array into an attribute dict of numpy arrays

    Parameters
    -----------
    array : SciDBArray
       An array with 1 or more non-string attributes

    compression : None, 1-9, or 'auto'
       Whether to use compression in the transfer

    Returns
    -------
    dict : att name -> numpy array
    """

    contents = array.interface._scan_array(array.name, fmt=_fmt(array), **kwargs)
    dtype = [(str(nm), null_typemap[t, nullable])
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
            result[nm] = np.where(good, att['data'], NULLS[typ]).astype(typemap[typ])

        if typ == 'datetimetz':
            result[nm] = result[nm]['time'] - result[nm]['tz']

    return result


def _attribute_dict(array, compression):
    """
    Download+parse an array into a dict of numpy array attributes
    """

    # for speed, evaluate a query if it contains strings and nonstrings
    s = [r[1] == 'string' for r in array.sdbtype.full_rep]
    if any(s) and (not all(s)):
        array.eval()

    # partition string and non-string attributes. Download and process
    # separately for speed

    atts = {}
    for isstring, dt in groupby(array.sdbtype.full_rep, lambda r: r[1] == 'string'):
        # project if a subset of attributes
        subatts = [nm for (nm, _, _) in dt]
        if len(subatts) < array.natt:
            subarray = array.project(*subatts)
        else:
            subarray = array

        if isstring:
            a = _string_attribute_dict(subarray, compression=compression)
        else:
            a = _nonstring_attribute_dict(subarray, compression=compression)
        atts.update(**a)

    return atts


def toarray_dense(array, compression='auto'):
    """
    Convert a dense SciDBArray to a numpy array

    Avoids unpacking() the array for speed.

    Warning
    -------
    This method will fail if any of the cells in the
    SciDB array are empty!
    """
    from .schema_utils import coerced_shape

    # determine shape and dtype of final result
    shp = coerced_shape(array)
    sz = np.product(shp)

    atts = _attribute_dict(array, compression)
    dtype = [(nm, atts[nm].dtype)
             for (nm, d, n) in array.datashape.sdbtype.full_rep]

    inds = _scidb_serialize(np.arange(sz).reshape(shp), array.datashape.chunk_size)
    result = np.empty(sz, dtype)

    for k in atts:
        if atts[k].size < sz:
            raise ValueError("Illegal dense download: array has empty cells")
        result[k][inds] = atts[k]

    result = result.reshape(shp)

    # For convenience:
    # cast single-attribute record arrays into plain numpy arrays
    if len(result.dtype) == 1:
        result = result[result.dtype.names[0]]

    return result


def toarray_sparse(array, compression='auto'):
    """
    Convert a SciDBArray to a numpy array.

    unpacks the array and explicitly downloads indices. This
    adds processing and memory overhead, but this is the only way
    to properly deal with sparsity in SciDB.

    Notes
    -----
    index 0 of output array is aligned with the lower bound
    of the scidb array.
    """

    unpacked = array.unpack()
    atts = _attribute_dict(unpacked, compression)

    # shift nonzero origins
    inds = tuple(atts[d] - lo for
                 d, lo in zip(array.dim_names,
                              array.datashape.dim_low))

    # determine shape and dtype of final result
    shp = array.shape
    if shp is None:  # unbound array
        shp = tuple([i.max() + 1 if i.size > 0 else 0 for i in inds])
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


def tosparse_scipy(array, sparse_fmt, compression='auto'):
    from scipy import sparse
    from .schema_utils import coerced_shape

    try:
        spmat = getattr(sparse, sparse_fmt + "_matrix")
    except AttributeError:
        raise ValueError("Invalid matrix format: "
                         "'{0}'".format(sparse_fmt))
    if array.ndim != 2:
        raise ValueError("only recarray sparse format is valid for arrays with ndim != 2.")

    if array.natt != 1:
        raise ValueError("only recarray sparse format is valid for arrays with natt != 1.")

    shp = coerced_shape(array)
    unpacked = array.unpack()
    atts = _attribute_dict(unpacked, compression)

    # shift nonzero origins
    inds = tuple(atts[d] - lo for
                 d, lo in zip(array.dim_names,
                              array.datashape.dim_low))

    data = atts[array.att_names[0]]
    arr = sparse.coo_matrix((data, inds), shape=shp)

    return spmat(arr)


def tosparse_recarray(array, compression='auto'):

    unpacked = array.unpack()
    return toarray_dense(unpacked, compression)


def toarray(array, compression='auto', method='sparse'):
    dispatch = dict(sparse=toarray_sparse, dense=toarray_dense)
    try:
        func = dispatch[method]
    except KeyError:
        valid_keys = ','.join(sorted(dispatch.keys()))
        raise ValueError("method must be one of %s: %s" %
                         (valid_keys, method))
    return func(array, compression=compression)


def tosparse(array, sparse_fmt='recarray', compression='auto'):
    if sparse_fmt == 'recarray':
        return tosparse_recarray(array, compression)
    return tosparse_scipy(array, sparse_fmt)
