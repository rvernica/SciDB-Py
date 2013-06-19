"""SciDB Array Wrapper"""
import numpy as np
import re


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
            self.full_dtype = [('x0', dtype)]
        else:
            self.full_dtype = dtype

        # If a single dimension, make dtype a simple type
        if len(self.full_dtype) == 1:
            self.dtype = self.full_dtype[0][1]
            
        # If not specified, make chunks have ~10^6 values
        if chunk_size is None:
            chunk_size = max(10, int(1E6 ** (1. / len(self.shape))))
        if not hasattr(chunk_size, '__len__'):
            chunk_size = [chunk_size for s in self.shape]
        if len(chunk_size) != len(self.shape):
            raise ValueError("length of chunk_size should match "
                         "number of dimensions")
        #self.chunk_size = [min(c, s) for c, s in zip(chunk_size, self.shape)]
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
        R = re.compile(r'(?P<arrname>[\s\S]+)\<(?P<dtypes>\S*?)\>(?:\s*)\[(?P<dshapes>\S+)\]')
        match = R.search(descr.lstrip('schema').strip())
        try:
            D = match.groupdict()
            arrname = D['arrname']
            dtypes = D['dtypes']
            dshapes = D['dshapes']
        except:
            raise ValueError("no match for descr: {0}".format(descr))
        

        # split dtypes.  TODO: how to represent NULLs?
        R = re.compile(r'(\S*?):([\S ]*?),')
        dtype = R.findall(dtypes + ',')  # note added trailing comma

        # split dshapes.  TODO: correctly handle '*' dimensions
        #                       handle non-integer dimensions?
        R = re.compile(r'(\S*?)=(\S*?):(\S*?),(\S*?),(\S*?),')
        dshapes = R.findall(dshapes + ',')  # note added trailing comma

        shape = [int(d[2]) - int(d[1]) + 1 for d in dshapes]
        dim_names = [d[0] for d in dshapes]
        chunk_size = [int(d[3]) for d in dshapes]
        chunk_overlap = [int(d[4]) for d in dshapes]
        return cls(shape, dtype,
                   dim_names=[d[0] for d in dshapes],
                   chunk_size=[int(d[3]) for d in dshapes],
                   chunk_overlap=[int(d[4]) for d in dshapes])

    @property
    def descr(self):
        type_arg = ','.join(['{0}:{1}'.format(name, typ)
                             for name, typ in self.full_dtype])
        shape_arg = ','.join(['i{0}=0:{1},{2},{3}'.format(i, s - 1, cs, co)
                              for i, (s, cs, co)
                              in enumerate(zip(self.shape,
                                               self.chunk_size,
                                               self.chunk_overlap))])
        return '<{0}> [{1}]'.format(type_arg, shape_arg)


class SciDBArray(object):
    def __init__(self, datashape, interface, name, persistent=False):
        self.datashape = datashape
        self.interface = interface
        self.name = name
        self.persistent = persistent

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

    def __del__(self):
        if not self.persistent:
            self.interface._delete_array(self.name)

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

        if dtype == 'double':
            # transfer bytes
            bytes_rep = self.interface._scan_array(self.name, n=0,
                                                   fmt='({0})'.format(dtype))
            arr = np.fromstring(bytes_rep, dtype=dtype).reshape(shape)
        else:
            # transfer ASCII
            dtype = np.dtype(dtype)
            str_rep = self.interface._scan_array(self.name, n=0, fmt='csv')
            arr = np.array(map(dtype.type, str_rep.strip().split('\n')[1:]),
                           dtype=dtype).reshape(shape)
        return arr
