from __future__ import print_function, absolute_import, unicode_literals

from .robust import cross_join
from .utils import interleave, as_list, _new_attribute_label
from . import schema_utils as su
from .scidbarray import SciDBArray


class Alias(SciDBArray):

    def __init__(self, other, alias):
        super(Alias, self).__init__(other.datashape, other.interface,
                                    other.name, other.persistent)
        self.__query_label__ = "%s AS %s" % (self.name, alias)


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

    if how != 'inner':
        raise NotImplementedError("Only inner joins are supported for now.")

    lnames = set(left.att_names) | set(left.dim_names)
    rnames = set(right.att_names) | set(right.dim_names)
    on = on or list(lnames & rnames)
    on = as_list(on)
    left_on = right_on = on
    both_on = set(left_on) & set(right_on)

    for l in left_on:
        if l not in lnames:
            raise ValueError("Left array join name is invalid: %s" % l)
    for r in right_on:
        if r not in rnames:
            raise ValueError("Right array join name is invalid: %s" % r)

    # fully disambiguate arrays
    left, right, left_on, right_on = _apply_suffix(left, right,
                                                   left_on, right_on,
                                                   suffixes)

    keep = (set(left.att_names) | set(left.dim_names)
            | set(right.dim_names) | set(right.att_names))
    keep = keep - set(b + suffixes[1] for b in both_on)

    # build necessary dimensions to join on
    # XXX push this logic into cross_join
    left, left_on = _prepare_join_schema(left, left_on)
    right, right_on = _prepare_join_schema(right, right_on)
    result = cross_join(left, right, *interleave(left_on, right_on))

    # throw away index dimensions added by _prepare_join_schema
    idx = _new_attribute_label('_row', result)
    result = result.unpack(idx)
    result = result.project(*(a for a in result.att_names if a in keep))

    # drop the suffix on both_on
    result = result.attribute_rename(*(item for b in both_on
                                     for item in (b + suffixes[0], b)))
    return result
