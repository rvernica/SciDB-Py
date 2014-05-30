__all__ = ['change_axis_schema']

import numpy as np


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

    if start is not None:
        raise NotImplementedError("start is not supported")
    names = list(datashape.dim_names)
    stops = list(datashape.shape)
    chunks = list(datashape.chunk_size)
    overlaps = list(datashape.chunk_overlap)
    if stop is not None:
        stops[axis] = stop + 1
    if chunk is not None:
        chunks[axis] = chunk
    if overlap is not None:
        overlaps[axis] = overlap
    if name is not None:
        names[axis] = name
    return SciDBDataShape(stops, datashape.dtype, dim_names=names,
                          chunk_size=chunks, chunk_overlap=overlaps)


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
