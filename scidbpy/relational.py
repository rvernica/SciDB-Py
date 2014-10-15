from __future__ import print_function, absolute_import, unicode_literals

from .robust import cross_join
from .utils import interleave, as_list, _new_attribute_label
from . import schema_utils as su


def _prepare_join_schema(array, on):
    """
    Prepare an array to be joined on the specified attributes or dimensions
    """
    new_on = list(on)

    dt = dict((nm, typ) for nm, typ, _ in array.sdbtype.full_rep)

    f = array.afl

    for i, att in enumerate(on):
        if att not in dt or 'int' in dt[att]:
            continue

        new_att = _new_attribute_label('%s_idx' % att, array)
        idx = f.uniq(f.sort(array[att])).eval()
        array = f.index_lookup(array.as_('L'),
                               idx,
                               'L.%s' % att,
                               new_att,
                               "'index_sorted=true'")
        new_on[i] = new_att

    result = su.to_dimensions(array, *new_on)
    return result, new_on


def _apply_suffix(left, right, left_on, right_on, suffixes):
    """
    Fully disambiguate left and right schemas by applying suffixes.

    Returns
    -------
    new_left, new_right, new_left_on, new_right_on
    """
    lnames = set(left.att_names) | set(left.dim_names)
    rnames = set(right.att_names) | set(right.dim_names)

    # add suffix to join column names
    left_on_old = list(left_on)
    left_on = [l if l not in right_on else l + suffixes[0]
               for l in left_on]
    right_on = [r if r not in left_on_old else r + suffixes[1]
                for r in right_on]

    duplicates = list(lnames & rnames)

    def _relabel(x, dups, suffix):
        x = x.attribute_rename(*(item for d in dups if d in x.att_names
                                 for item in (d, d + suffix)))
        x = x.dimension_rename(*(item for d in dups if d in x.dim_names
                                 for item in (d, d + suffix)))
        return x

    return (_relabel(left, duplicates, suffixes[0]),
            _relabel(right, duplicates, suffixes[1]),
            left_on, right_on)


def merge(left, right, on=None, left_on=None, right_on=None,
          how='inner', suffixes=('_x', '_y')):
    """
    Perform a pandas-like join on two SciDBArrays.

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
       The new SciDB array. The new array has a single dimension,
       and an attribute for each attribute + dimension in left and right.

    Examples
    --------



    Notes
    -----
    This function wraps the SciDB cross_join operator, but performs several
    preprocessing steps::

      - Attributes are converted into dimensions automatically
      - Chunk size and overlap is standardized
      - Joining on non-integer attributes is handled using index_lookup
    """

    lnames = set(left.att_names) | set(left.dim_names)
    rnames = set(right.att_names) | set(right.dim_names)

    if how != 'inner':
        raise NotImplementedError("Only inner joins are supported for now.")

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
        on = on or list(lnames & rnames)
        on = as_list(on)
        left_on = right_on = on

    for l in left_on:
        if l not in lnames:
            raise ValueError("Left array join name is invalid: %s" % l)
    for r in right_on:
        if r not in rnames:
            raise ValueError("Right array join name is invalid: %s" % r)

    # fully disambiguate arrays
    left_on_orig = left_on
    left, right, left_on, right_on = _apply_suffix(left, right,
                                                   left_on, right_on,
                                                   suffixes)

    keep = (set(left.att_names) | set(left.dim_names)
            | set(right.dim_names) | set(right.att_names))
    keep = keep - set(right_on)

    # build necessary dimensions to join on
    # XXX push this logic into cross_join
    left, left_on = _prepare_join_schema(left, left_on)
    right, right_on = _prepare_join_schema(right, right_on)

    result = cross_join(left, right, *interleave(left_on, right_on))

    # throw away index dimensions added by _prepare_join_schema
    idx = _new_attribute_label('_row', result)
    result = result.unpack(idx)
    result = result.project(*(a for a in result.att_names if a in keep))

    # drop suffixes if they aren't needed
    renames = []
    for a in result.att_names:
        if not a.endswith(suffixes[0]):
            continue

        if a.replace(suffixes[0], suffixes[1]) not in result.att_names:
            renames.extend((a, a.replace(suffixes[0], '')))

    result = result.attribute_rename(*renames)
    return result