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

__all__ = ['join', 'merge']

from .schema_utils import change_axis_schema


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
        for i in range(len(ds.dim_names)):
            ds = change_axis_schema(ds, i, chunk=target.chunk_size[i],
                                    overlap=target.chunk_overlap[i])
        if a.datashape.schema != ds.schema:
            a = a.redimension(ds.schema)
        result.append(a)

    return result


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
