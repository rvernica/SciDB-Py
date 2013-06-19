"""SciDB Array Wrapper"""


class SciDBDataShape(object):
    """Object to store SciDBArray data type and shape"""
    def __init__(self, shape, dtype, dim_names=None,
                 chunk_size=None, chunk_overlap=0):
        try:
            self.shape = tuple(shape)
        except:
            self.shape = (shape,)

        self.dtype = dtype

        # TODO: allow numpy-style dtype declarations
        if type(dtype) is str:
            self.full_dtype = [('x0', dtype)]
        else:
            self.full_dtype = dtype

        if chunk_size is None:
            chunk_size = min(10, int(10000 ** (1. / len(self.shape))))
        if not hasattr(chunk_size, '__len__'):
            chunk_size = [chunk_size for s in self.shape]
        if len(chunk_size) != len(self.shape):
            raise ValueError("length of chunk_size should match "
                         "number of dimensions")
        self.chunk_size = [min(c, s) for c, s in zip(chunk_size, self.shape)]

        if not hasattr(chunk_overlap, '__len__'):
            chunk_overlap = [chunk_overlap for s in self.shape]
        if len(chunk_overlap) != len(self.shape):
            raise ValueError("length of chunk_overlap should match "
                             "number of dimensions")
        self.chunk_overlap = chunk_overlap

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

    def __del__(self):
        if not self.persistent:
            self.interface._delete_array(self.name)
