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

__all__ = ['join', 'merge', 'gemm', 'cumulate',
           'reshape', 'gesvd', 'thin']

from .schema_utils import change_axis_schema
from .utils import _new_attribute_label


def assert_single_attribute(array):
    if len(array.att_names) > 1:
        raise ValueError("Array must have a single attribute: %s" % array.name)


def as_same_dimension(*arrays):
    """
    Coerce arrays into the same shape if possible, or raise a ValueError

    Parameters
    ----------
    *arrays: One or more arrays
        The arrays to coerce

    Returns
    -------
    new_arrays : tuple of SciDBArrays
    """
    ndim = arrays[0].ndim
    for a in arrays:
        if a.ndim == ndim:
            continue
        # XXX could try broadcasting here
        raise ValueError("Invalid array dimensions: %s vs %s" % (ndim, a.ndim))
    return arrays


def _att_match(rep, item):
    """ Test of a single item in a full sdbtype rep matches another full_rep """

    # ignore nullability component
    return any(r[:2] == item[:2] for r in rep)


def _find_rename(rep, item, reserved):
    """
    Search a sdbtype full_rep for an attribute which matches
    another attribute in everything but name

    Parameters
    ----------
    rep : sdbtype.full_rep
    item: row from sdbtype.full_rep
    reserved: list of reserved attribute names

    Returns
    -------
    A name from rep, that matches item in all but
    name, and whose name doesn't appear in the reserved list.

    If no such item exist, returns None
    """

    for nm, tp, nul in rep:
        if nm in reserved:
            continue
        if (tp, nul) == item[1:]:
            return nm


def match_attribute_names(*arrays):
    """
    Rename attributes in an array list as necessary, to match all names

    Parameters
    ----------
    *arrays : one or more SciDBArrays

    Returns
    -------
    arrays: tuple of SciDBArrays
       All output arrays have the same attribute names

    Raises
    ------
    ValueError: if arrays aren't conformable
    """
    rep = arrays[0].sdbtype.full_rep
    result = [arrays[0]]
    for a in arrays[1:]:
        renames = []
        reserved = list(a.att_names)  # reserved att names
        for r in a.sdbtype.full_rep:
            nm = r[0]
            if _att_match(rep, r):
                reserved.append(nm)
                continue
            newname = _find_rename(rep, r, reserved)
            if newname is None:
                raise ValueError("Cannot rename %s in %s" % (nm, a))
            renames.extend((nm, newname))
            reserved.append(newname)
        if renames:
            a = a.afl.attribute_rename(a, *renames)
        result.append(a)
    return result


def match_chunks(*arrays):
    """
    Redimension arrays so they have identical chunk sizes and overlaps

    It is assumed that all input arrays have the same dimensionality.
    """
    target = arrays[0].datashape
    result = []
    for a in arrays:
        ds = a.datashape
        for i, j in zip(reversed(list(range(a.ndim))),
                        reversed(list(range(target.ndim)))):
            ds = change_axis_schema(ds, i, chunk=target.chunk_size[j],
                                    overlap=target.chunk_overlap[j])
        if a.datashape.schema != ds.schema:
            a = a.redimension(ds.schema)
        result.append(a)

    return result


def match_chunk_permuted(src, target, indices):
    """
    Rechunk an array to match a target, along a set of
    permuted dimensions

    Parameters
    ----------
    src : SciDBArray
        The array to modify
    target: SciDBArray
        The array to match
    indices: A list of tuples
        Each tuple (i,j) indicates that
        dimension *j* of src should have the same chunk properties
        as dimension *i* of target

    Returns
    -------
    A (possibly redimensioned) version of src
    """

    ds = src.datashape.copy()
    for i, j in indices:
        ds.chunk_size[j] = target.datashape.chunk_size[i]
        ds.chunk_overlap[j] = target.datashape.chunk_overlap[i]

    if ds.schema != src.datashape.schema:
        src = src.redimension(ds.schema)

    return src


def rechunk(array, chunk_size=None, chunk_overlap=None):
    """
    Change the chunk size and/or overlap

    Parameters
    ----------
    array : SciDBArray
        The array to sanitize
    chunk_size : int or list of ints (optional)
       The new chunk_size. Defaults to old chunk_size
    chunk_overlap: int or list of ints (optional)
       The new chunk overlap. Defaults to old chunk overlap

    Returns
    -------
    Either array (if unmodified) or a redimensioned version of array
    """

    # deal with chunk sizes
    ds = array.datashape.copy()
    if chunk_size is None:
        chunk_size = ds.chunk_size
    if isinstance(chunk_size, int):
        chunk_size = [chunk_size] * ds.ndim
    ds.chunk_size = chunk_size

    if chunk_overlap is None:
        chunk_overlap = ds.chunk_overlap
    if isinstance(chunk_overlap, int):
        chunk_overlap = [chunk_overlap] * ds.ndim
    ds.chunk_overlap = chunk_overlap

    if ds != array.datashape:
        array = array.redimension(ds.schema)
    return array


def boundify(array):
    """
    Redimension an array as needed so that no dimension
    is unbound (ie ending with *)
    """
    if not any(d is None for d in array.datashape.dim_high):
        return array

    ds = array.datashape.copy()
    idx = _new_attribute_label('_', array)
    bounds = array.unpack(idx).max().toarray()
    ds.dim_high = list(ds.dim_high)
    for i in range(array.ndim):
        if ds.dim_high[i] is not None:
            continue
        ds.dim_high[i] = int(bounds['%s_max' % ds.dim_names[i]][0])

    if ds != array.datashape:
        array = array.redimension(ds.schema)

    return array


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

    a, b = as_same_dimension(a, b)
    a, b = match_chunks(a, b)
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
    a, b = as_same_dimension(a, b)
    a, b = match_chunks(a, b)
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
    a = rechunk(a, chunk_size=chunk_size, chunk_overlap=0)
    b = rechunk(b, chunk_size=chunk_size, chunk_overlap=0)
    c = rechunk(c, chunk_size=chunk_size, chunk_overlap=0)
    a = boundify(a)
    b = boundify(b)
    c = boundify(c)
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
    array = rechunk(array, chunk_size=32, chunk_overlap=0)
    return array.afl.gesvd(array, *args)


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
    array = rechunk(array, chunk_overlap=0)
    return array.afl.cumulate(array, *args)


def reshape(array, *args):
    """
    Robust AFL reshape
    """
    array = rechunk(array, chunk_overlap=0)
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
        ds.chunk_size[i] = ds.chunk_size[i] / step * step

    if ds != array.datashape:
        array = array.redimension(ds.schema)

    return array.afl.thin(array, *args)
