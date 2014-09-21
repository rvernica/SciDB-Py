import numpy as np
from .utils import _new_attribute_label


def assert_single_attribute(array):
    """
    If the array has one attribute, do nothing. Else, raise ValueError
    """
    if len(array.att_names) > 1:
        raise ValueError("Array must have a single attribute: %s" % array.name)
    return array


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


def match_chunk_permuted(src, target, indices, match_bounds=False):
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
    match_bounds : bool (optional, default False)
        If true, match the chunk dimensions as well

    Returns
    -------
    A (possibly redimensioned) version of src
    """

    ds = src.datashape.copy()
    ds.dim_low = list(ds.dim_low)
    ds.dim_high = list(ds.dim_high)

    for i, j in indices:
        if not isinstance(i, int):
            i = target.dim_names.index(i)
        if not isinstance(j, int):
            j = src.dim_names.index(j)
        ds.chunk_size[j] = target.datashape.chunk_size[i]
        ds.chunk_overlap[j] = target.datashape.chunk_overlap[i]
        if match_bounds:
            ds.dim_low[j] = target.datashape.dim_low[i]
            ds.dim_high[j] = target.datashape.dim_high[i]

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
    return SciDBDataShape(None, datashape.sdbtype, dim_names=names,
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

    The first array is guaranteed *not* to be modified
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
    print(array.schema, dimensions, attributes)
    if array.dim_names == dimensions and array.att_names == attributes:
        return array

    orig_atts = set(array.att_names)
    orig_dims = set(array.dim_names)

    to_promote = [d for d in dimensions if d in orig_atts]  # att->dim
    to_demote = [a for a in attributes if a in orig_dims]  # dim->att

    # need a dummy attribute, otherwise result has no attributes
    if not attributes:
        dummy = _new_attribute_label('__dummy', array)
        array = array.apply(dummy, 0)
        attributes = [dummy]

    # build the attribute schema
    new_att = {}
    for r in array.sdbtype.full_rep:
        if r[0] in attributes:  # copy schema
            new_att[r[0]] = _att_schema_item(r)
    for d in to_demote:  # change attribute to dimension
        new_att[d] = '%s:int' % d

    new_att = ','.join(new_att[a] for a in attributes)

    # build the dimension schema
    ds = array.datashape
    new_dim = {}
    for n, l, h, ch, co in zip(ds.dim_names, ds.dim_low, ds.dim_high,
                               ds.chunk_size, ds.chunk_overlap):
        if n in dimensions:
            new_dim[n] = '{0}={1}:{2},{3},{4}'.format(n, l, h, ch, co)

    if to_promote:
        for k, v in limits(array, to_promote).items():
            new_dim[k] = _dim_schema_item(k, v)

    new_dim = ','.join(new_dim[d] for d in dimensions)

    schema = '<{0}> [{1}]'.format(new_att, new_dim)
    return array.redimension(schema)


def match_size(*arrays):
    """
    Resize all arrays in a list to the size of the first array
    """
    target = arrays[0].datashape
    result = []

    for a in arrays:
        ds = a.datashape.copy()
        ds.dim_low = list(ds.dim_low)
        ds.dim_high = list(ds.dim_high)

        for i in range(min(a.ndim, target.ndim)):
            ds.dim_low[i] = target.dim_low[i]
            ds.dim_high[i] = target.dim_high[i]
        if ds != a.datashape:
            a = a.redimension(ds.schema)
        result.append(a)

    return result


def expand(*arrays):
    """
    Grow arrays to equal shape, without truncating any data
    """
    arrays = list(map(boundify, arrays))
    assert_schema(arrays, same_dimension=True)

    dim_low = list(map(min, zip(*(a.datashape.dim_low for a in arrays))))
    dim_high = list(map(max, zip(*(a.datashape.dim_high for a in arrays))))

    result = []
    for a in arrays:
        ds = a.datashape.copy()
        ds.dim_low = dim_low
        ds.dim_high = dim_high
        if ds != a.datashape:
            a = a.redimension(ds.schema)
        result.append(a)

    return result


def as_column_vector(array):
    """
    Convert a 1D array into a 2D array with a single column
    """
    idx = _new_attribute_label('idx', array)
    ds = array.datashape.copy()
    ds.dim_low = list(ds.dim_low) + [0]
    ds.dim_high = list(ds.dim_high) + [0]
    ds.chunk_size = list(ds.chunk_size) * 2
    ds.chunk_overlap = list(ds.chunk_overlap) * 2
    ds.dim_names = list(ds.dim_names) + [idx]
    return array.redimension(ds.schema)


def as_row_vector(array):
    """
    Convert a 1D array into a 2D array with a single row
    """
    idx = _new_attribute_label('idx', array)
    ds = array.datashape.copy()
    ds.dim_low = [0] + list(ds.dim_low)
    ds.dim_high = [0] + list(ds.dim_high)
    ds.chunk_size = list(ds.chunk_size) * 2
    ds.chunk_overlap = list(ds.chunk_overlap) * 2
    ds.dim_names = [idx] + list(ds.dim_names)
    return array.redimension(ds.schema)


def zero_indexed(array):
    """
    Redimension an array so all lower coordinates are at 0

    Will only grow an array that starts above zero. If the
    array has any dimensions starting below zero,
    will raise a ValueError
    """
    if all(dl == 0 for dl in array.datashape.dim_low):
        return array
    if any(dl < 0 for dl in array.datashape.dim_low):
        raise ValueError("Cannot zero_index array: one or more "
                         "dimensions start < 0")

    ds = array.datashape.copy()
    ds.dim_low = [0] * ds.ndim
    return array.redimension(ds.schema)


def match_dimensions(A, B, dims):
    """
    Match the dimension bounds along a list of dimensions in 2 arrays.

    Parameters
    ----------
    A : SciDBArray
        First array
    B : SciDBArray
        Second array
    dims : list of pairs of integers
        For each (i,j) pair, indicates that A[i] should have same
        dimension boundaries as B[j]

    Returns
    -------
    Anew, Bnew : SciDBArrays
        (Possibly redimensioned) versions of A and B
    """
    dsa = A.datashape.copy()
    dsb = B.datashape.copy()
    dsa.dim_low = list(dsa.dim_low)
    dsb.dim_low = list(dsb.dim_low)
    dsa.dim_high = list(dsa.dim_high)
    dsb.dim_high = list(dsb.dim_high)

    for i, j in dims:
        low = min(dsa.dim_low[i], dsb.dim_low[j])
        high = max(dsa.dim_high[i], dsb.dim_high[j])

        dsa.dim_low[i] = low
        dsa.dim_high[i] = high
        dsb.dim_low[j] = low
        dsb.dim_high[j] = high

    if dsa != A.datashape:
        A = A.redimension(dsa.schema)
    if dsb != B.datashape:
        B = B.redimension(dsb.schema)

    return A, B


def right_dimension_pad(array, n):
    """
    Add dummy dimensions as needed to an array, so that it is at least n-dimensional.
    """
    if array.ndim >= n:
        return array

    atts = [_new_attribute_label('_dim%i' % i, array) for i in range(n - array.ndim)]
    apply_args = [x for item in enumerate(atts) for x in item[::-1]]
    return redimension(array.apply(*apply_args), array.dim_names + atts, array.att_names)


def left_dimension_pad(array, n):
    """
    Add dummy dimensions as needed to an array, so that it is at least n-dimensional.
    """
    if array.ndim >= n:
        return array

    atts = [_new_attribute_label('_dim%i' % i, array) for i in range(n - array.ndim)]
    apply_args = [x for item in enumerate(atts) for x in item[::-1]]
    return redimension(array.apply(*apply_args), atts + array.dim_names, array.att_names)


def assert_schema(arrays, zero_indexed=False, bounded=False,
                  same_attributes=False, same_dimension=False):

    ds0 = arrays[0].datashape
    if same_dimension:
        if not all(a.ndim == ds0.ndim for a in arrays):
            raise ValueError("Input arrays must all have same dimension")

    for a in arrays:
        ds = a.datashape
        if zero_indexed and not all(dl == 0 for dl in ds.dim_low):
            raise ValueError("Input arrays must start at 0 "
                             "along all dimensions")
        if bounded and not all(dh is not None for dh in ds.dim_high):
            raise ValueError("Input arrays must be bound along "
                             "all dimensions")
        if same_attributes and ds.sdbtype.full_rep != ds0.sdbtype.full_rep:
            raise ValueError("Input arrays must have the same attributes")
