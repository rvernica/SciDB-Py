"""Functions to perform arithmetic on arrays"""

# License: Simplified BSD, 2014
# See LICENSE.txt for more information

from .robust import merge, join
from .utils import _new_attribute_label


def zero_fill(a, b):
    """
    Build a new array equal to a, but where cells
    non-empty cells in b are assigned to 0
    """
    # assumpiton: a, b are single attribute arrays
    att = _new_attribute_label('x', a, b)
    zeros = b.apply(att, '%s (0)' % a.sdbtype.full_rep[0][1])
    zeros = zeros.project(att).cast(a.schema)
    return merge(a, zeros)


def nan_fill(a):
    """Fill all empty cells in a with NaNs

    Parameters
    ----------
    a : SciDBArray
        The array to nan-fill

    Returns
    -------
    b : SciDBArray
        The nan-filled verion of a
    """

    # assumption: a is a single attribute array
    b = a.afl.build(a.datashape.schema, 'nan')
    return merge(a, b)


def nullify(a):
    """
    Make all attributes in an array nullable

    Copies the array if modifications are needed, else
    returns the input unchanged

    Parameters
    ----------
    a : SciDBArray
        The array to nullify

    Returns
    -------
    b : SciDBArray
       a if all attributes are nullable, else a nullable version of a
    """

    from .scidbarray import sdbtype

    rep = list(map(list, a.sdbtype.full_rep))
    for r in rep:
        r[2] = True

    newtype = sdbtype.from_full_rep(rep)
    if newtype != a.sdbtype:
        return a.afl.cast(a, "%s %s" % (newtype, a.datashape.dim_schema))
    return a


def assert_single_attribute(a):
    if len(a.att_names) != 1:
        raise ValueError("Array must have a single attribute")


def sparse_join(a, b, op):
    """
    Peform a [op] b, where op is an arithmetic operator, and a and/or b are sparse.
    Empty cells are treated as zero.

    Parameters
    ----------
    op : A SciDB AFL operator
       E.g., afl.add
    a : SciDBArray
       Left operand
    b : SciDBArray
       Right operand

    Returns
    -------
    out : SciDBArray
        A new SciDBArray object
    """
    assert_single_attribute(a)
    assert_single_attribute(b)

    a = nullify(a)
    b = nullify(b)

    _a = zero_fill(a, b)
    _b = zero_fill(b, a)
    a, b = _a, _b

    f = a.afl
    attr = _new_attribute_label('f0', a, b)

    fill_with_nans = op.__name__ in ['div', 'mod']

    op = op(a.att_names[0], b.att_names[0])
    result = f.papply(join(a, b), attr, op)
    if fill_with_nans:
        result = nan_fill(result)

    return result
