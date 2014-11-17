# License: Simplified BSD, 2014
# See LICENSE.txt for more information

"""
Database-style joins
"""
from __future__ import absolute_import, print_function, division, unicode_literals
import logging

from .robust import cross_join, join
from .utils import interleave, as_list, _new_attribute_label
from . import schema_utils as su


def _prepare_categories(left, right, left_on, right_on):
    """
    Search join predicates for attributes, compute categorical
    index dimensions, build new join lists

    Returns new versions of inputs
    """
    f = left.afl

    new_left = list(left_on)
    new_right = list(right_on)

    for i, (l, r) in enumerate(zip(left_on, right_on)):
        if l in left.dim_names and r in right.dim_names:
            continue

        # XXX handle case where only one is attribute

        cats = f.sort(left[l]).uniq().eval()
        l_cat = _new_attribute_label('%s_cat' % l, left)
        r_cat = _new_attribute_label('%s_cat' % r, right)

        new_left[i] = l_cat
        new_right[i] = r_cat
        left = left.index_lookup(cats, l, l_cat)
        right = right.index_lookup(cats, r, r_cat)

    left = su.to_dimensions(left, *new_left)
    right = su.to_dimensions(right, *new_right)
    return left, right, new_left, new_right


def _disambiguate(array, avoid, joins, suffix):
    """
    Resolve name collisions

    Returns a new copy of the array
    """
    to_rename = ([a for a in array.att_names if a in avoid] +
                 [d for d in array.dim_names if d in avoid
                  and d not in joins])

    renames = dict((x, x + suffix) for x in to_rename)
    return array.relabel(renames)


def _validate_ons(left, right, on, left_on, right_on):
    """
    Check that join predicates are valid

    Returns valid left_on, right_on lists
    """

    lnames = set(left.att_names) | set(left.dim_names)
    rnames = set(right.att_names) | set(right.dim_names)

    if (left_on is not None or right_on is not None) and on is not None:
        raise ValueError("Cannot specify left_on/right_on with on")

    if left_on is not None or right_on is not None:
        if left_on is None or right_on is None:
            raise ValueError("Must specify both left_on and right_on")

        left_on = as_list(left_on)
        right_on = as_list(right_on)
        if len(left_on) != len(right_on):
            raise ValueError("left_on and right_on must have "
                             "the same number of items")

    else:
        # default join is on matching dimensions
        on = on or list(set(left.dim_names) & set(right.dim_names))
        on = as_list(on)
        left_on = right_on = on

    for l in left_on:
        if l not in lnames:
            raise ValueError("Left array join name is invalid: %s" % l)
    for r in right_on:
        if r not in rnames:
            raise ValueError("Right array join name is invalid: %s" % r)

    for l, r in zip(left_on, right_on):
        # we don't currently handle dimension-attribute joins
        if (l in left.att_names) ^ (r in right.att_names):
            raise NotImplementedError("Attribute-dimension join pair not supported: "
                                      " %s, %s" % (l, r))

    return left_on, right_on


def merge(left, right, on=None, left_on=None, right_on=None,
          how='inner', suffixes=('_x', '_y')):
    """
    Perform a Pandas-like join on two SciDBArrays.

    Parameters
    ----------
    left : SciDBArray
       The left array to join on
    right : SciDBArray
       The right array to join on
    on : None, string, or list of strings
       The names of dimensions or attributes to join on. Either
       on or both `left_on` and `right_on` must be supplied.
       If on is supplied, the specified names must exist in both
       left and right
    left_on : None, string, or list of strings
        The names of dimensions or attributes in the left array to join on.
        If provided, then right_on must also be provided, and have as many
        elements as left_on
    right_on : None, string, or list of strings
        The name of dimensions or attributes in the right array to join on.
        See notes above for left_join
    how : 'inner' | 'left' | 'right' | 'outer'
        The kind of join to perform. Currently, only 'inner' is supported.
    suffixes : tuple of two strings
        The suffix to add to array dimensions or attributes which
        are duplicated in left and right.

    Returns
    -------
    joined : SciDBArray
       The joined array.

    Notes
    -----
    When joining on attributes, a categorical index is computed
    for each array. This index will appear as a dimension in the output.

    This function builds an AFL join or cross join query,
    performing preprocessing on the inputs as necessary to match chunk
    sizes, avoid name collisions, etc.

    If neither on, left_on, or right_on are provided, then the join
    defaults to the overlapping dimension names.
    """
    if how != 'inner':
        raise NotImplementedError("Only inner joins are supported for now.")

    lnames = set(left.att_names) | set(left.dim_names)
    rnames = set(right.att_names) | set(right.dim_names)

    left_on, right_on = _validate_ons(left, right, on, left_on, right_on)

    logging.getLogger(__name__).debug("Joining on %s, %s" % (left_on, right_on))

    # turn attributes into categorical dimensions
    # XXX need to update this when joins besides inner supported
    left, right, left_on, right_on = _prepare_categories(left, right, left_on, right_on)

    # shortcut: if we are joining on all dimensions of both arrays, use
    # a join instead of a cross join. SciDB is much faster with this operation
    lidx = [left.dim_names.index(l) for l in left_on]
    ridx = [right.dim_names.index(r) for r in right_on]
    if lidx == ridx and len(lidx) == left.ndim and len(ridx) == right.ndim:
        return join(left, right)

    if (list(left_on) == list(left.dim_names)) and (list(right_on) == list(right.dim_names)):
        return join(left, right)

    # add suffixes to disambiguate non-join columns
    # scidb throws out the redundant join columns for us, so no need
    # to rename
    left = _disambiguate(left, rnames, left_on, suffixes[0])
    right = _disambiguate(right, lnames, right_on, suffixes[1])

    return cross_join(left, right, *interleave(left_on, right_on))
