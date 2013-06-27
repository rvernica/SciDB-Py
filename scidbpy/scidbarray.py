"""SciDB Array Wrapper"""
import numpy as np
import re
from .errors import SciDBError


class SciDBDataShape(object):
    """Object to store SciDBArray data type and shape"""
    def __init__(self, shape, dtype, dim_names=None,
                 chunk_size=None, chunk_overlap=0):
        try:
            self.shape = tuple(shape)
        except:
            self.shape = (shape,)

        self.dtype = dtype

        # TODO: make dtypes play well with numpy
        if type(dtype) is str:
            self.full_dtype = [('x0', dtype, '')]
        else:
            self.full_dtype = dtype

        # If a single dimension, make dtype a simple type
        if len(self.full_dtype) == 1:
            self.dtype = self.full_dtype[0][1]

        if dim_names is None:
            dim_names = ['i{0}'.format(i) for i in range(len(self.shape))]
        if len(dim_names) != len(self.shape):
            raise ValueError("length of dim_names should match "
                             "number of dimensions")
        self.dim_names = dim_names

        # If not specified, make chunks have ~10^6 values
        if chunk_size is None:
            chunk_size = max(10, int(1E6 ** (1. / len(self.shape))))
        if not hasattr(chunk_size, '__len__'):
            chunk_size = [chunk_size for s in self.shape]
        if len(chunk_size) != len(self.shape):
            raise ValueError("length of chunk_size should match "
                         "number of dimensions")
        self.chunk_size = chunk_size

        if not hasattr(chunk_overlap, '__len__'):
            chunk_overlap = [chunk_overlap for s in self.shape]
        if len(chunk_overlap) != len(self.shape):
            raise ValueError("length of chunk_overlap should match "
                             "number of dimensions")
        self.chunk_overlap = chunk_overlap

    @classmethod
    def from_descr(cls, descr):
        """Create a DataShape object from a SciDB Descriptor.

        This function uses a series of regular expressions to take an input
        descriptor such as::

            descr = "not empty myarray<val:double> [i0=0:3,4,0,i1=0:4,5,0]"

        parse it, and return a SciDBDataShape object.
        """
        # First split out the array name, data types, and shapes.
        # e.g. "myarray<val:double> [i=0:4,5,0]"
        #      => arrname = "myarray"; dtypes="val:double"; dshapes="i=0:9,5,0"
        R = re.compile(r'(?P<arrname>[\s\S]+)\<(?P<dtypes>[\S\s]*?)\>(?:\s*)'
                       '\[(?P<dshapes>\S+)\]')
        descr = descr.lstrip('schema').strip()
        match = R.search(descr)
        try:
            D = match.groupdict()
            arrname = D['arrname']
            dtypes = D['dtypes']
            dshapes = D['dshapes']
        except:
            raise ValueError("no match for descr: {0}".format(descr))

        #if 'NULL' in dtypes:
        #    raise NotImplementedError("Nullable dtypes: {0}".format(dtypes))

        # split dtypes.  TODO: how to represent NULLs?
        R = re.compile(r'(\S*?):([\S ]*?)\s?(NULL)?,')
        dtypes = R.findall(dtypes + ',')  # note added trailing comma

        if len(dtypes) > 1:
            raise NotImplementedError("Compound dtypes: {0}".format(dtypes))

        # split dshapes.  TODO: correctly handle '*' dimensions
        #                       handle non-integer dimensions?
        R = re.compile(r'(\S*?)=(\S*?):(\S*?),(\S*?),(\S*?),')
        dshapes = R.findall(dshapes + ',')  # note added trailing comma

        return cls(shape=[int(d[2]) - int(d[1]) + 1 for d in dshapes],
                   dtype=dtypes,
                   dim_names=[d[0] for d in dshapes],
                   chunk_size=[int(d[3]) for d in dshapes],
                   chunk_overlap=[int(d[4]) for d in dshapes])

    @property
    def descr(self):
        type_arg = ','.join(['{0}:{1} {2}'.format(name, typ, null)
                             for name, typ, null in self.full_dtype])
        shape_arg = ','.join(['{0}=0:{1},{2},{3}'.format(d, s - 1, cs, co)
                              for (d, s, cs, co) in zip(self.dim_names,
                                                        self.shape,
                                                        self.chunk_size,
                                                        self.chunk_overlap)])
        return '<{0}> [{1}]'.format(type_arg, shape_arg)


class SciDBAttribute(object):
    """
    A simple class to reference SciDB attributes,
    i.e. things with names in the SciDB database instance.
    """
    def __init__(self, name):
        self.name = name

    @staticmethod
    def parse(obj):
        """Parse object for insertion into a query.

        If the object is a SciDBAttribute, the name attribute is returned.
        Otherwise, the object itself is returned.
        """
        if isinstance(obj, SciDBAttribute):
            return obj.name
        else:
            return obj


class SciDBIndexLabel(SciDBAttribute):
    def __init__(self, arr, i):
        self.arr = arr
        self.i = i

    @property
    def name(self):
        return self.arr.datashape.dim_names[self.i]


class SciDBValLabel(SciDBAttribute):
    def __init__(self, arr, i):
        self.arr = arr
        self.i = i

    @property
    def name(self):
        return "{0}.{1}".format(self.arr.name,
                                self.arr.datashape.full_dtype[self.i][0])


class SciDBArray(SciDBAttribute):
    def __init__(self, datashape, interface, name, persistent=False):
        self._datashape = datashape
        self.interface = interface
        self.name = name
        self.persistent = persistent

    @property
    def datashape(self):
        if self._datashape is None:
            try:
                schema = self.interface._show_array(self.name, fmt='csv')
                self._datashape = SciDBDataShape.from_descr(schema)
            except SciDBError:
                self._datashape = None
        return self._datashape

    @property
    def shape(self):
        return self.datashape.shape

    @property
    def ndim(self):
        return len(self.datashape.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def dtype(self):
        return self.datashape.dtype

    def index(self, i):
        """Return a SciDBAttribute representing the i^th index"""
        return SciDBIndexLabel(self, i)

    def val(self, i):
        """Return a SciDBAttribute representing the i^th value in each cell"""
        return SciDBValLabel(self, i)

    def __del__(self):
        if (self.datashape is not None) and (not self.persistent):
            self.interface.query("remove({0})", self)

    def __repr__(self):
        show = self.interface._show_array(self.name, fmt='csv').split('\n')
        return "SciDBArray({0})".format(show[1])

    def contents(self, n=0):
        return repr(self) + '\n' + self.interface._scan_array(self.name, n)

    def toarray(self):
        # TODO: use bytes for formats other than double
        # TODO: correctly handle compound dtypes
        # TODO: correctly handle nullable values
        dtype = self.datashape.dtype
        shape = self.datashape.shape

        if dtype[-1][-1] == "NULL":
            raise NotImplementedError("Nullable dtypes")

        if dtype == 'double':
            # transfer bytes
            bytes_rep = self.interface._scan_array(self.name, n=0,
                                                   fmt='({0})'.format(dtype))
            arr = np.fromstring(bytes_rep, dtype=dtype).reshape(shape)
        else:
            # transfer ASCII
            dtype = np.dtype(dtype[:2])
            str_rep = self.interface._scan_array(self.name, n=0, fmt='csv')
            arr = np.array(map(dtype.type, str_rep.strip().split('\n')[1:]),
                           dtype=dtype).reshape(shape)
        return arr

    def __getitem__(self, slices):
        # Note that slice steps must be a divisor of the chunk size.
        # TODO: handle non-slice indices
        if len(slices) < self.ndim:
            slices = list(slices) + [slice(None)
                                     for i in range(self.ndim - len(slices))]
        if len(slices) != self.ndim:
            raise ValueError("too many indices")

        indices = [sl.indices(sh) for sl, sh in zip(slices, self.shape)]

        # TODO: do this more efficiently: is subarray needed? is thin needed?
        #       remove tmp array?
        limits = [i[0] for i in indices] + [i[1] - 1 for i in indices]
        steps = sum([[0, i[2]] for i in indices], [])
        
        tmp = self.interface.new_array()
        arr = self.interface.new_array()
        self.interface.query("store(subarray({0},{2}),{1})",
                             self, tmp,
                             SciDBAttribute(','.join(str(L) for L in limits)))
        self.interface.query("store(thin({0},{2}),{1})",
                             tmp, arr,
                             SciDBAttribute(','.join(str(st) for st in steps)))
        return arr

    # note that these operations only work across the first attribute
    # in each array.
    def _join_operation(self, other, op):
        if isinstance(other, SciDBArray):
            if self.shape != other.shape:
                raise NotImplementedError("array shapes must match")
            arr = self.interface.new_array()
            self.interface.query("store(project(apply(join({0},{1}),"
                                 "x,{2}{3}{4}),x),{5})", self, other,
                                 self.val(0), op, other.val(0), arr)
            return arr
        elif np.isscalar(other):
            arr = self.interface.new_array()
            self.interface.query("store(project(apply({0},"
                                 "x,{1}{2}{3}),x),{4})",
                                 self, self.val(0), op, other, arr)
            return arr
        else:
            raise ValueError("unrecognized value: {0}".format(other))

    def __add__(self, other):
        return self._join_operation(other, '+')

    def __sub__(self, other):
        return self._join_operation(other, '-')

    def __mul__(self, other):
        return self._join_operation(other, '*')

    def __div__(self, other):
        return self._join_operation(other, '/')

    def __mod__(self, other):
        return self._join_operation(other, '%')
