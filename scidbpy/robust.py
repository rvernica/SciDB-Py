"""
The functions in this module closely correspond to AFL calls,
but they preprocess arrays as needed to satisfy any requirements
that SciDB imposes on the schemas of arrays used as AFL arguments.

As an example, the AFL merge function requires that its two input
arrays have the same attribute list, number of dimensions, and dimension
start index. The merge() function performs this preprocessing as needed.
"""

# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from __future__ import absolute_import, print_function, division, unicode_literals

__all__ = ['join', 'merge', 'gemm', 'cumulate',
           'reshape', 'gesvd', 'thin', 'cross_join', 'uniq']

from .utils import _new_attribute_label, interleave, new_alias_label
from . import schema_utils as su


def merge(a, b):
    """
    Robust AFL merge operation

    Parameters
    ----------
    a : SciDBArray
        Left array
    b : SciDBArray
        Right array

    Notes
    -----
    Performs any of the following steps if needed:

    * Broadcast A and B to equal shapes
    * Match attribute names, if unambiguous
    * Align array origins

    Returns
    -------
    result : SciDBArray
       merge(a, b)
    """

    # From AFL docs:
    # The two arrays should have the same attribute list, number of
    # dimensions, and dimension start index. If the dimensions are not
    # the same size, the output array uses the larger of the two.
    a, b = su.as_same_dimension(a, b)
    a, b = su.expand(a, b)
    a, b = su.match_chunks(a, b)
    return a.afl.merge(a, b)


def join(a, b):
    """
    Robust AFL join operation

    Parameters
    ----------
    a : SciDBArray
        Left array
    b : SciDBArray
        Right array

    Notes
    -----
    Performs any of the following steps if needed:

    * Broadcast A and B to equal shapes
    * Align array origins
    * Match chunk sizes and overlaps

    Returns
    -------
    result : SciDBArray
       join(a, b)
    """

    # From AFL docs:
    # The two arrays must have the same dimension start coordinates,
    # the same chunk size, and the same chunk overlap

    # XXX match start coordinates here
    a, b = su.as_same_dimension(a, b)
    a, b = su.match_chunks(a, b)
    return a.afl.join(a, b)


def gemm(a, b, c):
    """
    Robust AFL gemm operation

    Performs a * b + c

    Redimensions inputs if necessary

    Parameters
    ----------
    a : SciDBArray
        First array
    b : SciDBArray
        Second array
    c : SciDBArray
        Third array

    Returns
    -------
    result : SciDBArray
        a * b + c
    """

    """ From docs

    The first attribute of all three arrays must be of type double. All other attributes are ignored.
    The chunks of the input matrices must be square, and must have a chunk interval between 32 and 1024.
    Each dimension of each matrix must have the following characteristics:
    Currently, the starting index must be zero.
    The ending index cannot be '*'.
    Currently, the chunk overlap must be zero.
    """
    for x in [a, b, c]:
        if x.sdbtype.full_rep[0][1] != 'double':
            raise TypeError("Matrix multiply requires a type double for first attribute.")

    chunk_size = min(max(a.datashape.chunk_size[0], 32), 1024)
    a = su.rechunk(a, chunk_size=chunk_size, chunk_overlap=0)
    b = su.rechunk(b, chunk_size=chunk_size, chunk_overlap=0)
    c = su.rechunk(c, chunk_size=chunk_size, chunk_overlap=0)
    a = su.boundify(a)
    b = su.boundify(b)
    c = su.boundify(c)
    return a.afl.gemm(a, b, c)


def gesvd(array, *args):
    """
    Robust AFL svd call

    Parameters
    ----------
    array : SciDBArray
    *args : Subsequent arguments to SVD AFL call

    Notes
    -----
    Rechunks array if needed by AFL
    """
    array = su.rechunk(array, chunk_size=32, chunk_overlap=0)
    return array.afl.gesvd(array, *args)

def _thin(array, *args):
        
    start = args[::2]
    step  = args[1::2]

    dim_names = [dimname.encode('ascii') for dimname in (array.dim_names)]
    origatt = array.att_names
    
    filterstr = ""
    for d,st,sp in zip(dim_names, start, step):  
        tempstr = " ({0} - {1}) % {2}=0 and".format(d,st, sp)
        filterstr+=tempstr  
    filterstr+=(" 1") #TODO: This is kind of lazy, just completes the and.Needs cleaner code
    
    applystr   = ""
    newdimlist = []
    for d,st,sp in zip(dim_names,start,step):  
        tempstr = "_{0}_copy,({0}/{2})-{1},".format(d,st,sp)
        newdimlist.append("_{0}_copy".format(d))
        applystr+=tempstr  
    applystr = applystr.rstrip(',')
    
    array = array.afl.filter(array, filterstr).apply(applystr)
    #array = array.afl.filter(array, filterstr)
    array = su.redimension(array, newdimlist, origatt)
    
    dimlist = newdimlist + dim_names
    dimlist[::2] = newdimlist
    dimlist[1::2] = dim_names
    array   = su.dimension_rename(array, *dimlist)
    
    return array

def cumulate(array, *args):
    """
    Robust AFL cumulate call

    Parameters
    ----------
    array : Array to cumulate
    *args: Additional arguments to pass to AFL cumulate()

    Returns
    -------
    cumulate(array, *args)

    Notes
    -----
    Re-chunks array to have no chunk overlap, if needed
    """
    array = su.rechunk(array, chunk_overlap=0)
    return array.afl.cumulate(array, *args)


def reshape(array, *args):
    """
    Robust AFL reshape
    """
    array = su.rechunk(array, chunk_overlap=0)
    return array.afl.reshape(array, *args)


def thin(array, *args):
    """
    Robust AFL thin call

    Parameters
    ----------
    array : SciDBArray
        The array to thin
    args: sequence of ints
        sequence of start, step for each dimension

    Notes
    -----
    The array is redimensioned if necessary, so that
    the chunk size is a multiple of the thin steps
    """

    ds = array.datashape.copy()

    # ensure step divides chunk_size
    for i, (start, step) in enumerate(zip(args[::2], args[1::2])):
        ds.chunk_size[i] = ds.chunk_size[i] // step * step

    if ds != array.datashape:
        array = array.redimension(ds.schema)

    return _thin(array, *args)


def cross_join(a, b, *dims):
    """
    Robust AFL cross_join

    Parameters
    ----------
    a : SciDBArray
       The left array in the join
    b : SciDBArray
       The right array in the join
    *dims : Pairs of dimension names
       The dimensions to join along

    Notes
    -----
    Arrays will be rechunked as needed for the cross join to run
    """

    adims = dims[::2]
    bdims = dims[1::2]

    # match chunk info of joined dimensions
    inds = tuple((i, j) for i, j in zip(adims, bdims))
    b, a = su.match_chunk_permuted(b, a, inds)

    # use aliases if needed
    if (any(d in a.dim_names for d in bdims) or
            any(d in b.dim_names for d in adims)):
        l = su.new_alias_label('L', a, b)
        r = su.new_alias_label('R', a, b)
        adims = ['%s.%s' % (l, d) for d in adims]
        bdims = ['%s.%s' % (r, d) for d in bdims]
        dims = interleave(adims, bdims)
        return a.afl.cross_join(a.as_(l), b.as_(r), *dims)

    return a.afl.cross_join(a, b, *dims)


def uniq(a, is_sorted=False):
    """
    Robust AFL uniq operator

    Parameters
    ----------
    a : SciDBArray
        Array to compute unique elements of. Must have a single attribute
    is_sorted: bool
        Whether the array is pre_sorted

    Returns
    -------
    u : SciDBArray
       The unique elements of A

    Notes
    -----
    Will flatten and/or sort the input as necessar
    """

    su.assert_single_attribute(a)

    # ravel if need be
    if a.ndim != 1 and is_sorted:
        raise ValueError("Cannot use is_sorted with multidimensional arrays")
    if a.ndim != 1:
        att = _new_attribute_label('idx')
        a = a.unpack(att)[a.att_names[0]]

    # sort if need be
    if not is_sorted:
        a = a.sort()
    return a.uniq()
