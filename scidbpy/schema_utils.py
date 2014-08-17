__all__ = ['change_axis_schema']

import numpy as np
from .utils import _new_attribute_label


def change_axis_schema(datashape, axis, start=None, stop=None,
                       chunk=None, overlap=None, name=None):
    """
    Create a new DataShape by modifying the parameters of one axis

    Parameters
    ----------
    datashape : SciDBDataShape
        The template data shape
    axis : int
        Which axis to modify
    stop : int (optional)
        New axis upper bound
    chunk : int (optional)
        New chunk size
    overlap : int (optional)
        New chunk overlap
    name : str (optional)
        New dimension name

    Returns
    -------
    A new SciDBDataShape, obtained by overriding the input parameters
    of the template datashape along the specified axis
    """
    from .scidbarray import SciDBDataShape

    names = list(datashape.dim_names)
    starts = list(datashape.dim_low)
    stops = list(datashape.dim_high)
    chunks = list(datashape.chunk_size)
    overlaps = list(datashape.chunk_overlap)
    if stop is not None:
        stops[axis] = stop
    if chunk is not None:
        chunks[axis] = chunk
    if overlap is not None:
        overlaps[axis] = overlap
    if name is not None:
        names[axis] = name
    if start is not None:
        starts[axis] = start
    shp = [stp - strt + 1 for strt, stp in zip(starts, stops)]
    return SciDBDataShape(shp, datashape.sdbtype, dim_names=names,
                          chunk_size=chunks, chunk_overlap=overlaps,
                          dim_low=starts, dim_high=stops)


def _unique(val, taken):
    if val not in taken:
        return val
    offset = 2
    while '%s_%i' % (val, offset) in taken:
        offset += 1
    return '%s_%i' % (val, offset)


def _rename_att(datashape, index, name):
    from scidbpy import SciDBDataShape, sdbtype

    atts = datashape.sdbtype.names
    if atts[index] == name:
        return datashape

    rep = [list(x) for x in datashape.sdbtype.full_rep]
    rep[index][0] = name
    rep = [tuple(x) for x in rep]

    schema = "tmp%s%s" % (sdbtype(np.dtype(rep)).schema, datashape.dim_schema)
    return SciDBDataShape.from_schema(schema)


def _rename_dim(datashape, index, name):
    from scidbpy import SciDBDataShape

    names = datashape.dim_names
    if names[index] == name:
        return datashape

    schema = "tmp" + datashape.schema
    # XXX doesn't work if fixing non-first duplicate dim name
    schema = schema.replace('%s=' % names[index], '%s=' % name, 1)
    return SciDBDataShape.from_schema(schema)


def disambiguate(*arrays):
    """
    Process a list of arrays with calls to cast as needed, to avoid
    any name collisions in dimensions or attributes
    """
    from .scidbarray import SciDBArray

    all_names = [name for a in arrays if isinstance(a, SciDBArray)
                 for nameset in [a.dim_names, a.att_names]
                 for name in nameset]

    # no collisions, return unmodified
    if len(set(all_names)) == len(all_names):
        return arrays

    taken = set()
    result = []
    afl = None

    for a in arrays:
        if not isinstance(a, SciDBArray):
            result.append(a)
            continue

        afl = afl or a.afl

        ds = a.datashape
        for i, att in enumerate(a.att_names):
            att = _unique(att, taken)
            taken.add(att)
            ds = _rename_att(ds, i, att)
        for i, dim in enumerate(a.dim_names):
            dim = _unique(dim, taken)
            taken.add(dim)
            ds = _rename_dim(ds, i, dim)
        if ds.schema == a.datashape.schema:
            result.append(a)
        else:
            result.append(afl.cast(a, ds.schema))
    return tuple(result)


def _att_schema_item(rep):
    name, typ, nullable = rep
    result = '{0}:{1}'.format(name, typ)
    if nullable:
        result = result + ' NULL DEFAULT null'
    return result


def _dim_schema_item(name, limit):
    return '{0}={1}:{2},1000,0'.format(name, limit[0], limit[1])


def limits(array, names):
    """
    Compute the lower/upper bounds for a set of attributes

    Parameters
    ----------
    array : SciDBArray
        The array to consider
    names : list of strings
        Names of attributes to consider

    Returns
    -------
    limits : dict mapping name->(lo, hi)
        Contains the minimum and maximum value for each attribute

    Notes
    -----
    This performs a full scan of the array
    """

    args = ['%s(%s)' % (f, n)
            for n in names
            for f in ['min', 'max']]
    result = array.afl.aggregate(array, *args).toarray()
    return dict((n, (int(result['%s_min' % n][0]), int(result['%s_max' % n][0])))
                for n in names)


def redimension(array, dimensions, attributes):
    """
    Redimension an array as needed, swapping and dropping
    attributes as needed

    Parameters
    ----------
    array: SciDBArray
        The array to redimension
    dimensions : list of strings
        The dimensions or attributes in array that should be dimensions
    attributes : list of strings
        The dimensions or attributes in array that should be attributes

    Notes
    -----
    - Only integer attributes can be listed as dimensions
    - If an attribute or dimension in the original array is not explicitly
      provided as an input, it is dropped
    - If all attributes are marked for conversion to dimensions
      a new dummy attribute is added to ensure a valid schema.

    Returns
    -------
    result : SciDBArray
       A new version of array, redimensioned as needed to
       ensure proper dimension/attribute schema.
    """

    to_promote = set(dimensions) & set(array.att_names)  # att->dim
    to_demote = set(attributes) & set(array.dim_names)  # dim->att

    if not to_promote and not to_demote:
        return array

    # need a dummy attribute, otherwise result has no attributes
    if (to_promote == set(array.att_names)) and (not to_demote):
        dummy = _new_attribute_label('__dummy', array)
        array = array.apply(dummy, 0)
        attributes = list(attributes) + [dummy]

    # build the attribute schema
    atts = [_att_schema_item(r)
            for r in array.sdbtype.full_rep
            if r[0] in attributes]
    atts.extend('%s:int' % d
                for d in to_demote)
    atts = ','.join(atts)

    # build the dimension schema
    ds = array.datashape
    dims = ['{0}={1}:{2},{3},{4}'.format(n, l, h, ch, co)
            for n, l, h, ch, co in
            zip(ds.dim_names, ds.dim_low, ds.dim_high,
                ds.chunk_size, ds.chunk_overlap)
            if n in dimensions]
    dims.extend(_dim_schema_item(k, v)
                for k, v in limits(array, to_promote).items())
    dims = ','.join(dims)

    schema = '<{0}> [{1}]'.format(atts, dims)
    return array.redimension(schema)
