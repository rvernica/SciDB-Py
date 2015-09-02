# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np

from .utils import _new_attribute_label, new_alias_label


def assert_single_attribute(array):
    """
    Raise a ValueError if an array has multiple attributes

    Parameters
    ----------
    array : SciDBArray
        The array to test

    Returns
    -------
    array : SciDBArray
        The input array

    Raises
    ------
    ValueError
        if array has multiple attributes
    """
    if len(array.att_names) > 1:
        raise ValueError("Array must have a single attribute: %s" % array.name)
    return array


def as_same_dimension(*arrays):
    """
    Coerce arrays into the same shape if possible, or raise a ValueError

    Parameters
    ----------
    *arrays
       One or more arrays to test

    Returns
    -------
    new_arrays : tuple of SciDBArrays

    Raises
    ------
    ValueError
        if arrays have mismatched dimensions, and cannot be
        coerced into the same shape.

    Notes
    -----
    Currently this function only checkes for mismatched dimensions,
    it is unable to fix them.
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
    Rename attributes in a set of arrays, so that all arrays
    have the same names of attributes

    Parameters
    ----------
    *arrays
       one or more SciDBArrays

    Returns
    -------
    arrays : tuple of SciDBArrays
       All output arrays have the same attribute names

    Raises
    ------
    ValueError : if arrays aren't conformable

    Notes
    -----
    An array's attributes will be renamed to match an attribute name
    in the first array, if the association is unambiguous. For example,
    consider two arrays with attribute schemas ``<a:int32, b:float>`` and
    ``<a:int32, c:float>``. The attribute ``c`` will be renamed to ``b``, since
    the datatypes match and there is no other ``b`` attribute.
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
            a = a.attribute_rename(a, *renames)
        result.append(a)
    return tuple(result)


def match_chunks(*arrays):
    """
    Redimension arrays so they have identical chunk sizes and overlaps

    It is assumed that all input arrays have the same dimensionality. If
    needed, use :func:`as_same_dimension` to ensure this.

    Parameters
    ----------
    *arrays
        One or more arrays to match

    Returns
    -------
    arrays : Tuple of SciDBArrays
        The chunk sizes and overlaps will be matched to the first input.

    See Also
    --------
    match_chunk_permuted()
        to match chunks along particular pairs of
        dimensions
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

    return tuple(result)


def match_chunk_permuted(src, target, indices, match_bounds=False):
    """
    Match chunks along a set of dimension pairs.

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
        If true, match the dimension boundaries as well

    Returns
    -------
    new_src, new_target : tuple of SciDBArrays
        A (possibly redimensioned) version of the inputs
    """

    ds = src.datashape.copy()
    ds.dim_low = list(ds.dim_low)
    ds.dim_high = list(ds.dim_high)
    ds_target = target.datashape.copy()
    ds_target.dim_low = list(ds_target.dim_low)
    ds_target.dim_high = list(ds_target.dim_high)

    hi1 = ds.dim_high
    hi2 = ds_target.dim_high

    # lookup array dounds if schema is unbound
    if match_bounds:
        if any(l is None for l in hi1):
            tops = src.unpack('_').max().toarray()
            hi1 = [int(tops['%s_max' % l][0]) for l in src.dim_names]
        if any(l is None for l in hi2):
            tops = target.unpack('_').max().toarray()
            hi2 = [int(tops['%s_max' % l][0]) for l in target.dim_names]

    for i, j in indices:
        if not isinstance(i, int):
            i = target.dim_names.index(i)
        if not isinstance(j, int):
            j = src.dim_names.index(j)
        ds.chunk_size[j] = target.datashape.chunk_size[i]
        ds.chunk_overlap[j] = target.datashape.chunk_overlap[i]
        if match_bounds:
            l = min(ds.dim_low[j], ds_target.dim_low[i])
            h = max(hi1[j], hi2[i])

            ds.dim_low[j] = l
            ds.dim_high[j] = h
            ds_target.dim_low[i] = l
            ds_target.dim_high[i] = h

    if ds.schema != src.datashape.schema:
        src = src.redimension(ds.schema)
    if ds_target.schema != target.datashape.schema:
        target = target.redimension(ds_target.schema)

    return src, target


def rechunk(array, chunk_size=None, chunk_overlap=None):
    """
    Change the chunk size and/or overlap

    Parameters
    ----------
    array : SciDBArray
        The array to sanitize
    chunk_size : int or list of ints (optional)
       The new chunk_size. Defaults to old chunk_size
    chunk_overlap : int or list of ints (optional)
       The new chunk overlap. Defaults to old chunk overlap

    Returns
    -------
    array : SciDBArray
       A (possibly redimensioned) version of the input
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


def boundify(array, trim=False):
    """
    Redimension an array as needed so that no dimension is unbound (ie ending with *)

    Parameters
    ----------
    array : SciDBArray
       The array to bound

    Returns
    -------
    array : SciDBArray
       A (possibly redimensioned) version of array

    Notes
    -----
    This forces evaluation of lazy arrays
    """
    if not any(d is None for d in array.datashape.dim_high):
        return array

    if trim:
        return _boundify_trim(array)

    # use special scan syntax to get current bounds. eval() required
    dims = array.eval().dimensions().project('low', 'high').toarray()
    ds = array.datashape.copy()
    ds.dim_low = tuple(dims['low'])
    ds.dim_high = tuple(dims['high'])

    array = array.redimension(ds.schema)

    return array


def coerced_shape(array):
    """
    Return an array shape, even if the array is unbound.

    If the array is unbound, the shape is guaranteed to contain the data

    Parameters
    ----------
    array : SciDBArray
        The array to lookup the shape for

    returns
    -------
    shape : tuple of ints
        The shape
    """
    if array.shape is not None:
        return array.shape

    r = array.eval().dimensions().project('low', 'high').toarray()
    shp = tuple(np.maximum(r['high'] - r['low'] + 1, 0))
    return shp


def _boundify_trim(array):
    # actually scan the array to find boundaries

    ds = array.datashape.copy()
    idx = _new_attribute_label('_', array)
    bounds = array.unpack(idx).max().toarray()
    ds.dim_high = list(ds.dim_high)
    for i in range(array.ndim):
        if ds.dim_high[i] is not None:
            continue
        ds.dim_high[i] = int(bounds['%s_max' % ds.dim_names[i]][0])

    if ds.schema != array.schema:
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
    new_schema : SciDBDataShape
       The new schema, obtained by overriding the input parameters
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


def dimension_rename(array, *args):
    old = args[::2]
    new = args[1::2]
    ds = array.datashape

    for o, n in zip(old, new):
        axis = array.dim_names.index(o)
        ds = change_axis_schema(ds, axis, name=n)

    if ds.schema != array.datashape.schema:
        array = array.cast(ds.schema)

    return array

def attribute_rename(array, *args):
        
    tuplenouni = [arg.encode('ascii') for arg in reversed(args)] 
    new = tuplenouni[::2]
    old = tuplenouni[1::2]

    string_list = new + array.att_names
    #string_list.sort(key = lambda s: len(s))
    out = []
    for s in string_list: 
        if not any([s in old]): 
            out.append(s)
            
    applystr = str(tuplenouni).replace("'", "").strip('[]').rstrip(',')
    projstr  = str(out).replace("'", "").strip('[]').rstrip(',')
    array    = array.apply(applystr).project(projstr)

    return array
    

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

    # py2 numpy doesn't like unicode here
    rep = [(str(r[0]), r[1]) for r in rep]

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

    Parameters
    ----------
    *arrays
        One or more arrays to process

    Returns
    -------
    arrays : tuple of SciDBArrays
        The (possibly recasted) inputs. None of the dimensions or
        attribute names match.
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


def cast_to_integer(array, attributes):
    """
    Cast a set of attributes in an array to integer datatypes.

    This is a useful preprocessing step before redimensioning attributes
    as dimensions
    """
    atts = array.att_names

    for nm, typ, null in array.sdbtype.full_rep:
        if nm not in attributes:
            continue
        if 'int' in typ:
            continue
        if typ == 'bool':
            x = _new_attribute_label('__cast', array)
            array = array.attribute_rename(nm, x).apply(nm, 'iif(%s, 1, 0)' % x)
            continue
        else:
            raise ValueError("Don't know how to turn %s to int64" % typ)

    return array.project(*atts)


def to_dimensions(array, *attributes):
    """
    Ensure that a set of attributes or dimensions are dimensions

    Parameters
    -----------
    array : SciDBArray
        The array to promote
    attributes : one or more strings
        Attribute names to promote. Dimension labels are ignored

    Returns
    -------
    promoted : SciDBArray
        A new array
    """
    dims = list(array.dim_names) + [a for a in attributes if a in array.att_names]
    atts = [a for a in array.att_names if a not in attributes]
    return redimension(array, dims, atts)


def to_attributes(array, *dimensions):
    """
    Ensure that a set of attributes or dimensions are attributes

    Parameters
    -----------
    array : SciDBArray
        The array to promote
    dimensions : one or more strings
        Dimension names to demote. Attribute labels are ignored

    Returns
    -------
    demoted : SciDBArray
        A new array
    """
    dims = [d for d in array.dim_names if d not in dimensions]
    atts = list(array.att_names) + [d for d in dimensions if d in array.dim_names]
    return redimension(array, dims, atts)


def redimension(array, dimensions, attributes, dim_boundaries=None):
    """
    Redimension an array as needed, swapping and dropping attributes as needed.

    Parameters
    ----------
    array: SciDBArray
        The array to redimension
    dimensions : list of strings
        The dimensions or attributes in array that should be dimensions
    attributes : list of strings
        The dimensions or attributes in array that should be attributes
    dim_boundaries : dict (optional)
        A dictionary mapping dimension names to boundary tuples (lo, hi)
        Specifies the dimension bounds for attributes promoted to dimensions.
        If not provided, will default to (0,*). WARNING: this will
        fail if promiting negatively-valued attributes to dimensions.

    Notes
    -----
    - Only integer attributes can be listed as dimensions
    - If an attribute or dimension in the original array is not explicitly
      provided as an input, it is dropped
    - If no attributes are specified,
      a new dummy attribute is added to ensure a valid schema.

    Returns
    -------
    result : SciDBArray
       A new version of array, redimensioned as needed to
       ensure proper dimension/attribute schema.
    """
    if array.dim_names == dimensions and array.att_names == attributes:
        return array
    dim_boundaries = dim_boundaries or {}

    orig_atts = set(array.att_names)
    orig_dims = set(array.dim_names)

    to_promote = [d for d in dimensions if d in orig_atts]  # att->dim
    to_demote = [a for a in attributes if a in orig_dims]  # dim->att
    array = cast_to_integer(array, to_promote)

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
        new_att[d] = '%s:int64' % d

    new_att = ','.join(new_att[a] for a in attributes)

    # build the dimension schema
    ds = array.datashape
    new_dim = {}
    for n, l, h, ch, co in zip(ds.dim_names, ds.dim_low, ds.dim_high,
                               ds.chunk_size, ds.chunk_overlap):
        h = h if h is not None else '*'
        if n in dimensions:
            new_dim[n] = '{0}={1}:{2},{3},{4}'.format(n, l, h, ch, co)

    if to_promote:
        # don't do limits here, too expensive!
        # XXX this does wrong thing if attribute has negative values
        # for k, v in limits(array, to_promote).items():
        for k in to_promote:
            v = dim_boundaries.get(k, (0, '*'))
            new_dim[k] = _dim_schema_item(k, v)

    new_dim = ','.join(new_dim[d] for d in dimensions)

    schema = '<{0}> [{1}]'.format(new_att, new_dim)
    return array.redimension(schema)


def match_size(*arrays):
    """
    Resize all arrays in a list to the size of the first array. This
    requires that all arrays span a subset of the first array's domain.

    Parameters
    ----------
    *arrays
       One or more SciDBArrays

    Returns
    -------
    arrays : tuple of SciDBArrays
       The (possibly redimensioned) inputs. All arrays are resized
       to match the first array

    Raises
    ------
    ValueError : If any arrays have a domain that is not a subset
                 of the first array's domain.

    """
    target = arrays[0].datashape
    result = []

    # check for bad inputs
    for a in arrays:
        ds = a.datashape.copy()
        for i in range(min(a.ndim, target.ndim)):
            if ds.dim_low[i] < target.dim_low[i] or \
                    ds.dim_high[i] > target.dim_high[i]:
                raise ValueError("All array domains must be a subset "
                                 "of the first array's domain")

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

    return tuple(result)


def expand(*arrays):
    """
    Grow arrays to equal shape, without truncating any data

    Parameters
    ----------
    *arrays
       One or more SciDBArrays

    Returns
    -------
    arrays : tuple of SciDBArrays
       The input arrays, redimensioned as needed so they all have
       the same domain.
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
    if array.ndim != 1:
        raise ValueError("Array must be 1D")

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
    if array.ndim != 1:
        raise ValueError("Array must be 1D")
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

    Raises
    -------
    ValueError : if any array has dimensions starting below zero.
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

    Parameters
    ----------
    array : SciDBArray
        The array to pad
    n : int
        The minimum dimensionality of the output

    Returns
    -------
    array : SciDBArray
       A version of the input, with extra dimensions added after the old
       dimensions.
    """
    if array.ndim >= n:
        return array

    nadd = n - array.ndim
    atts = [_new_attribute_label('_dim%i' % i, array) for i in range(nadd)]
    apply_args = [x for item in enumerate(atts) for x in item[::-1]]

    ds = array.datashape.copy()
    ds.dim_low = list(ds.dim_low) + ([0] * nadd)
    ds.dim_high = list(ds.dim_high) + ([0] * nadd)
    ds.dim_names = list(ds.dim_names) + atts
    ds.chunk_overlap = list(ds.chunk_overlap) + ([0] * nadd)
    ds.chunk_size = list(ds.chunk_size) + ([1000] * nadd)

    return array.apply(*apply_args).redimension(ds.schema)


def left_dimension_pad(array, n):
    """
    Add dummy dimensions as needed to an array, so that it is at least n-dimensional.

    Parameters
    ----------
    array : SciDBArray
        The array to pad
    n : int
        The minimum dimensionality of the output

    Returns
    -------
    array : SciDBArray
       A version of the input, with extra dimensions added before the old
       dimensions.
    """
    if array.ndim >= n:
        return array
    nadd = n - array.ndim
    atts = [_new_attribute_label('_dim%i' % i, array) for i in range(nadd)]
    apply_args = [x for item in enumerate(atts) for x in item[::-1]]

    ds = array.datashape.copy()
    ds.dim_low = ([0] * nadd) + list(ds.dim_low)
    ds.dim_high = ([0] * nadd) + list(ds.dim_high)
    ds.dim_names = atts + list(ds.dim_names)
    ds.chunk_overlap = ([0] * nadd) + list(ds.chunk_overlap)
    ds.chunk_size = ([1000] * nadd) + list(ds.chunk_size)

    return array.apply(*apply_args).redimension(ds.schema)


def assert_schema(arrays, zero_indexed=False, bounded=False,
                  same_attributes=False, same_dimension=False):
    """
    Check that a set of arrays obeys a set of criteria on their schemas.

    Parameters
    ----------
    arrays : tuple of SciDBArrays
        The arrays to check

    zero_indexed : boolean, optional (default False)
        If True, check at all arrays have origins at 0
    bounded : boolean, optional (default False)
        If True, check that all arrays are bounded (ie don't have ``*`` in
                                                    the dimension schema)
    same_attributes : boolean, optional (default False)
       If True, check that all arrays have identical attribute
       names, order, datatypes, and nullability
    same_dimension : boolean, optional (default True)
       If True, check that all arrays have the same dimensionality

    Raises
    ------
    ValueError : If any test fails

    Returns
    -------
    arrays : tuple of SciDBArrays
        The input
    """

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

    return tuple(arrays)


def _relabel(array, renames):
    """
    renames is a dict mapping old names to new names
    """

    att_renames = []
    dim_renames = []

    for k, v in renames.items():
        if k in array.att_names:
            att_renames.extend([k, v])
        elif k in array.dim_names:
            dim_renames.extend([k, v])
        else:
            raise ValueError("Invalid array attribute: %s" % k)

    return array.attribute_rename(*att_renames).dimension_rename(*dim_renames)
