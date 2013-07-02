"""SciDB Array Wrapper"""
import numpy as np
import re
from .errors import SciDBError

__all__ = ["sdbtype", "SciDBArray", "SciDBDataShape"]


# Create mappings between scidb and numpy string representations
SDB_TYPE_LIST = ['bool', 'float', 'double',
                 'int8', 'int16', 'int32', 'int64',
                 'uint8', 'uint16', 'uint32', 'uint64']

# TODO: this uses system endian-ness.  We need to make sure that the local
#       big/little end properties match those of the cluster, or else our
#       data transfer will be corrupted

# convert type strings to numpy type identifiers
SDB_NP_TYPE_MAP = dict((typ, np.dtype(typ).descr[0][1])
                       for typ in SDB_TYPE_LIST)
NP_SDB_TYPE_MAP = dict((val, key) for key, val in SDB_NP_TYPE_MAP.iteritems())


class sdbtype(object):
    """Class to store info on SciDB types

    also contains conversion tools to and from numpy dtypes"""
    def __init__(self, typecode):
        print "-----------"
        print typecode
        print "-----------"
        # process array typecode: either a scidb descriptor or a numpy dtype
        if isinstance(typecode, sdbtype):
            self.descr = typecode.descr
            self.dtype = typecode.dtype
            self.full_rep = [t.copy() for t in typecode.full_rep]
        else:
            try:
                self.dtype = np.dtype(typecode)
                self.descr = None
            except:
                self.descr = self._regularize(typecode)
                self.dtype = None

            if self.dtype is None:
                self.dtype = self._descr_to_dtype(self.descr)
        
            if self.descr is None:
                self.descr = self._dtype_to_descr(self.dtype)

            self.full_rep = self._descr_to_list(self.descr)

    def __repr__(self):
        return "sdbtype('{0}')".format(self.descr)

    def __str__(self):
        return self.descr

    @classmethod
    def _regularize(cls, descr):
        # if a full descriptor is given, we need to split out the dtype
        R = re.compile(r'\<(?P<descr>[\S\s]*?)\>')
        results = R.search(descr)
        try:
            descr = results.groupdict()['descr']
        except AttributeError:
            pass
        return '<{0}>'.format(descr)

    @classmethod
    def _descr_to_list(cls, descr):
        """Convert a scidb type descriptor to a list representation

        Parameters
        ----------
        descr : string
            a SciDB type descriptor, for example
            "<val:double,rank:int32>"

        Returns
        -------
        sdbt_list : list of tuples
            the correspo
        """
        descr = cls._regularize(descr)
        descr = descr.lstrip('<').rstrip('>')

        # assume that now we just have the dtypes themselves
        # TODO: support default values?
        sdbL = descr.split(',')
        sdbL = [map(str.strip, s.split(':')) for s in sdbL]
        sdbL = [(s[0], s[1].split()[0], ('NULL' in s[1])) for s in sdbL]

        return sdbL

    @classmethod
    def _descr_to_dtype(cls, descr):
        """Convert a scidb type descriptor to a numpy dtype

        Parameters
        ----------
        descr : string
            a SciDB type descriptor, for example
            "<val:double,rank:int32>"

        Returns
        -------
        dtype : np.dtype object
            The corresponding numpy dtype descriptor
        """
        sdbL = cls._descr_to_list(descr)

        # TODO: support NULLs - this changes the memory layout, adding a byte
        #       for now, we'll just throw an error.
        #if np.any([s[2] for s in sdbL]):
        #    raise ValueError("NULL-able dtypes not supported")

        if len(sdbL) == 1:
            return np.dtype(sdbL[0][1])
        else:
            return np.dtype([(s[0], SDB_NP_TYPE_MAP[s[1]]) for s in sdbL])

    @classmethod
    def _dtype_to_descr(cls, dtype):
        """Convert a scidb type descriptor to a numpy dtype

        Parameters
        ----------
        dtype : np.dtype object
            The corresponding numpy dtype descriptor

        Returns
        -------
        descr : string
            a SciDB type descriptor, for example "<val:double,rank:int32>"
        """
        dtype = np.dtype(dtype).descr

        # Hack: if we re-encode this as a dtype, then de-encode again, numpy
        #       will add default names where they are missing
        dtype = np.dtype(dtype).descr
        pairs = ["{0}:{1}".format(d[0], NP_SDB_TYPE_MAP[d[1]]) for d in dtype]
        return '<{0}>'.format(','.join(pairs))
    

class SciDBDataShape(object):
    """Object to store SciDBArray data type and shape"""
    def __init__(self, shape, typecode, dim_names=None,
                 chunk_size=32, chunk_overlap=0):
        # Process array shape
        try:
            self.shape = tuple(shape)
        except:
            self.shape = (shape,)

        # process array typecode: either a scidb descriptor or a numpy dtype
        self.sdbtype = sdbtype(typecode)
        self.dtype = self.sdbtype.dtype

        # process array dimension names; define defaults if needed
        if dim_names is None:
            dim_names = ['i{0}'.format(i) for i in range(len(self.shape))]
        if len(dim_names) != len(self.shape):
            raise ValueError("length of dim_names should match "
                             "number of dimensions")
        self.dim_names = dim_names

        # process chunk sizes.  Either an integer, or a list of integers
        if not hasattr(chunk_size, '__len__'):
            chunk_size = [chunk_size for s in self.shape]
        if len(chunk_size) != len(self.shape):
            raise ValueError("length of chunk_size should match "
                         "number of dimensions")
        self.chunk_size = chunk_size

        # process chunk overlaps.  Either an integer, or a list of integers
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
        # First split out the array name, data types, and shapes.  e.g. 
        #
        #   "myarray<val:double,rank:int32> [i=0:4,5,0,j=0:9,5,0]"
        #
        # will become
        #
        #   dict(arrname = "myarray"
        #        dtypes  = "val:double,rank:int32"
        #        dshapes = "i=0:9,5,0,j=0:9,5,0")
        #
        R = re.compile(r'(?P<arrname>[\s\S]+)\<(?P<descr>[\S\s]*?)\>(?:\s*)'
                       '\[(?P<dshapes>\S+)\]')
        descr = descr.lstrip('schema').strip()
        match = R.search(descr)
        try:
            D = match.groupdict()
            arrname = D['arrname']
            descr = D['descr']
            dshapes = D['dshapes']
        except:
            raise ValueError("no match for descr: {0}".format(descr))

        # split dshapes.  TODO: correctly handle '*' dimensions
        #                       handle non-integer dimensions?
        R = re.compile(r'(\S*?)=(\S*?):(\S*?),(\S*?),(\S*?),')
        dshapes = R.findall(dshapes + ',')  # note added trailing comma

        return cls(shape=[int(d[2]) - int(d[1]) + 1 for d in dshapes],
                   typecode=descr,
                   dim_names=[d[0] for d in dshapes],
                   chunk_size=[int(d[3]) for d in dshapes],
                   chunk_overlap=[int(d[4]) for d in dshapes])

    @property
    def descr(self):
        shape_arg = ','.join(['{0}=0:{1},{2},{3}'.format(d, s - 1, cs, co)
                              for (d, s, cs, co) in zip(self.dim_names,
                                                        self.shape,
                                                        self.chunk_size,
                                                        self.chunk_overlap)])
        return '{0} [{1}]'.format(self.sdbtype, shape_arg)


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
    def __init__(self, arr, i, full=True):
        self.arr = arr
        self.i = i
        self.full = full

    @property
    def name(self):
        if self.full:
            return "{0}.{1}".format(self.arr.name,
                                    self.arr.datashape.dim_names[self.i])
        else:
            return self.arr.datashape.dim_names[self.i]


class SciDBValLabel(SciDBAttribute):
    def __init__(self, arr, i, full=True):
        self.arr = arr
        self.i = i
        self.full = full

    @property
    def name(self):
        if self.full:
            return "{0}.{1}".format(self.arr.name,
                                    self.arr.datashape.sdbtype.full_rep[self.i][0])
        else:
            return self.arr.datashape.sdbtype.full_rep[self.i][0]


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
    def sdbtype(self):
        return self.datashape.sdbtype

    @property
    def dtype(self):
        return self.datashape.dtype

    def index(self, i, full=True):
        """Return a SciDBAttribute representing the i^th index"""
        return SciDBIndexLabel(self, i, full)

    def val(self, i, full=True):
        """Return a SciDBAttribute representing the i^th value in each cell"""
        return SciDBValLabel(self, i, full)

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
        print self.datashape

        dtype = self.datashape.dtype
        shape = self.datashape.shape

        #if 'NULL' in str(self.datashape.sdbtype):
        #    raise NotImplementedError("NULL-able data types")

        if dtype == np.dtype('double'):
            # transfer bytes
            bytes_rep = self.interface._scan_array(self.name, n=0,
                                                   fmt='(double)')
            arr = np.fromstring(bytes_rep, dtype=dtype).reshape(shape)
        else:
            # transfer ASCII
            # XXX this is broken for compound data types
            # XXX need to parse commas
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

    def transpose(self):
        arr = self.interface.new_array()
        self.interface.query("store(transpose({0}), {1})",
                             self, arr)
        return arr

    T = property(transpose)
