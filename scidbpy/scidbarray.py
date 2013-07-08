"""SciDB Array Wrapper"""
import numpy as np
import re
import cStringIO
import warnings
from .errors import SciDBError

__all__ = ["sdbtype", "SciDBArray", "SciDBDataShape"]


# Create mappings between scidb and numpy string representations
_np_typename = lambda s: np.dtype(s).descr[0][1]
SDB_NP_TYPE_MAP = {'bool':_np_typename('bool'),
                   'float':_np_typename('float32'),
                   'double':_np_typename('float64'),
                   'int8':_np_typename('int8'),
                   'int16':_np_typename('int16'),
                   'int32':_np_typename('int32'),
                   'int64':_np_typename('int64'),
                   'uint8':_np_typename('uint8'),
                   'uint16':_np_typename('uint16'),
                   'uint32':_np_typename('uint32'),
                   'uint64':_np_typename('uint64'),
                   'char':_np_typename('c')}

NP_SDB_TYPE_MAP = dict((val, key) for key, val in SDB_NP_TYPE_MAP.iteritems())

SDB_IND_TYPE = 'int64'


class sdbtype(object):
    """SciDB data type class.

    This class encapsulates the information about the datatype of SciDB
    arrays, with tools to convert to and from numpy dtypes.

    Parameters
    ----------
    typecode : string, list, sdbtype, or dtype
        An object representing a datatype.  
    """
    def __init__(self, typecode):
        if isinstance(typecode, sdbtype):
            self.schema = typecode.schema
            self.dtype = typecode.dtype
            self.full_rep = [t.copy() for t in typecode.full_rep]

        else:
            try:
                self.dtype = np.dtype(typecode)
                self.schema = None
            except:
                self.schema = self._regularize(typecode)
                self.dtype = None

            if self.schema is None:
                self.schema = self._dtype_to_schema(self.dtype)

            if self.dtype is None:
                self.dtype = self._schema_to_dtype(self.schema)

            self.full_rep = self._schema_to_list(self.schema)

    def __repr__(self):
        return "sdbtype('{0}')".format(self.schema)

    def __str__(self):
        return self.schema

    @property
    def names(self):
        return [f[0] for f in self.full_rep]

    @property
    def bytes_fmt(self):
        """The format used for transfering raw bytes to and from scidb"""
        return '({0})'.format(','.join(rep[1] for rep in self.full_rep))
        

    @classmethod
    def _regularize(cls, schema):
        # if a full schema is given, we need to split out the type info
        R = re.compile(r'\<(?P<schema>[\S\s]*?)\>')
        results = R.search(schema)
        try:
            schema = results.groupdict()['schema']
        except AttributeError:
            pass
        return '<{0}>'.format(schema)

    @classmethod
    def _schema_to_list(cls, schema):
        """Convert a scidb type schema to a list representation

        Parameters
        ----------
        schema : string
            a SciDB type schema, for example
            "<val:double,rank:int32>"

        Returns
        -------
        sdbtype_list : list of tuples
            the corresponding full-representation of the scidbtype
        """
        schema = cls._regularize(schema)
        schema = schema.lstrip('<').rstrip('>')

        # assume that now we just have the dtypes themselves
        # TODO: support default values?
        sdbL = schema.split(',')
        sdbL = [map(str.strip, s.split(':')) for s in sdbL]
        sdbL = [(s[0], s[1].split()[0], ('NULL' in s[1])) for s in sdbL]

        return sdbL

    @classmethod
    def _schema_to_dtype(cls, schema):
        """Convert a scidb type schema to a numpy dtype

        Parameters
        ----------
        schema : string
            a SciDB type schema, for example
            "<val:double,rank:int32>"

        Returns
        -------
        dtype : np.dtype object
            The corresponding numpy dtype descriptor
        """
        sdbL = cls._schema_to_list(schema)

        if len(sdbL) == 1:
            return np.dtype(sdbL[0][1])
        else:
            return np.dtype([(s[0], SDB_NP_TYPE_MAP[s[1]]) for s in sdbL])

    @classmethod
    def _dtype_to_schema(cls, dtype):
        """Convert a scidb type schema to a numpy dtype

        Parameters
        ----------
        dtype : np.dtype object
            The corresponding numpy dtype descriptor

        Returns
        -------
        schema : string
            a SciDB type schema, for example "<val:double,rank:int32>"
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

        # process array typecode: either a scidb schema or a numpy dtype
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
    def from_schema(cls, schema):
        """Create a DataShape object from a SciDB Schema.

        This function uses a series of regular expressions to take an input
        schema such as::

            schema = "not empty myarray<val:double> [i0=0:3,4,0,i1=0:4,5,0]"

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
        R = re.compile(r'(?P<arrname>[\s\S]+)\<(?P<schema>[\S\s]*?)\>(?:\s*)'
                       '\[(?P<dshapes>\S+)\]')
        schema = schema.lstrip('schema').strip()
        match = R.search(schema)
        try:
            D = match.groupdict()
            arrname = D['arrname']
            schema = D['schema']
            dshapes = D['dshapes']
        except:
            raise ValueError("no match for schema: {0}".format(schema))

        # split dshapes.  TODO: correctly handle '*' dimensions
        #                       handle non-integer dimensions?
        R = re.compile(r'(\S*?)=(\S*?):(\S*?),(\S*?),(\S*?),')
        dshapes = R.findall(dshapes + ',')  # note added trailing comma

        return cls(shape=[int(d[2]) - int(d[1]) + 1 for d in dshapes],
                   typecode=schema,
                   dim_names=[d[0] for d in dshapes],
                   chunk_size=[int(d[3]) for d in dshapes],
                   chunk_overlap=[int(d[4]) for d in dshapes])

    @property
    def schema(self):
        shape_arg = ','.join(['{0}=0:{1},{2},{3}'.format(d, s - 1, cs, co)
                              for (d, s, cs, co) in zip(self.dim_names,
                                                        self.shape,
                                                        self.chunk_size,
                                                        self.chunk_overlap)])
        return '{0} [{1}]'.format(self.sdbtype, shape_arg)

    @property
    def ind_attr_dtype(self):
        """Construct a numpy dtype that can hold indices and values.

        This is useful in downloading sparse array data from SciDB.
        """
        keys = self.dim_names + self.sdbtype.names
        dct = dict(f[:2] for f in self.sdbtype.full_rep)
        types = [SDB_NP_TYPE_MAP[dct.get(key, SDB_IND_TYPE)]
                 for key in keys]
        return np.dtype(zip(keys, types))


class ArrayAlias(object):
    """
    An alias object used for constructing queries
    """
    def __init__(self, arr, name=None):
        self.arr = arr
        if name is None:
            self.name = self.arr.name
        else:
            self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __getattr__(self, attr):
        match = re.search(r'([da])(\d*)(f?)', attr)

        if not match:
            # exception text copied from Python2.6
            raise AttributeError("%r object has no attribute %r" %
                         (type(self).__name__, attr))
        
        groups = match.groups()
        i = int(groups[1])

        if groups[2]:
            # seeking a full, qualified name
            ret_str = self.arr.name + '.'
        else:
            ret_str = ''

        if groups[0] == 'd':
            # looking for a dimension name
            try:
                dim_name = self.arr.datashape.dim_names[i]
            except IndexError:
                raise ValueError("dimension index %i is out of bounds" % i)
            return ret_str + dim_name
            
        else:
            # looking for an attribute name
            try:
                attr_name = self.arr.datashape.sdbtype.full_rep[i][0]
            except IndexError:
                raise ValueError("attribute index %i is out of bounds" % i)
            return ret_str + attr_name


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
        if isinstance(obj, SciDBArray):
            return ArrayAlias(obj)
        elif isinstance(obj, SciDBAttribute):
            return obj.name
        else:
            return obj


class SciDBIndexLabel(SciDBAttribute):
    def __init__(self, arr, i, full=True, check_collision=None):
        self.arr = arr
        self.i = i
        self.full = full
        self.check_collision = check_collision

    @property
    def name(self):
        dim_name = self.arr.datashape.dim_names[self.i]
        if dim_name == self.parse(self.check_collision):
            dim_name = "{0}_2".format(dim_name)
        if self.full:
            return "{0}.{1}".format(self.arr.name, dim_name)
        else:
            return dim_name


class SciDBValLabel(SciDBAttribute):
    def __init__(self, arr, i, full=True):
        self.arr = arr
        self.i = i
        self.full = full

    @property
    def name(self):
        if self.full:
            full_rep = self.arr.datashape.sdbtype.full_rep[self.i][0]
            return "{0}.{1}".format(self.arr.name, full_rep)
                                    
        else:
            return self.arr.datashape.sdbtype.full_rep[self.i][0]


def _new_attribute_label(suggestion='val', *arrays):
    """Return a new attribute label

    The label will not clash with any attribute labels in the given arrays
    """
    label_list = sum([[dim[0] for dim in arr.sdbtype.full_rep]
                      for arr in arrays], [])
    if suggestion not in label_list:
        return suggestion
    else:
        # find all labels of the form val_0, val_1, val_2 ... etc.
        # where `val` is replaced by suggestion
        R = re.compile(r'^{0}_(\d+)$'.format(suggestion))
        nums = sum([map(int, R.findall(label)) for label in label_list], [])

        nums.append(-1)  # in case it's empty
        return '{0}_{1}'.format(suggestion, max(nums) + 1)


class SciDBArray(SciDBAttribute):
    def __init__(self, datashape, interface, name, persistent=False):
        self._datashape = datashape
        self.interface = interface
        self.name = name
        self.persistent = persistent

    def alias(self, name=None):
        return ArrayAlias(self, name)

    @property
    def datashape(self):
        if self._datashape is None:
            try:
                schema = self.interface._show_array(self.name, fmt='csv')
                self._datashape = SciDBDataShape.from_schema(schema)
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

    def index(self, i, **kwargs):
        """Return a SciDBAttribute representing the i^th index"""
        return SciDBIndexLabel(self, i, **kwargs)

    def val(self, i, **kwargs):
        """Return a SciDBAttribute representing the i^th value in each cell"""
        return SciDBValLabel(self, i, **kwargs)

    def __del__(self):
        if (self.datashape is not None) and (not self.persistent):
            self.interface.query("remove({0})", self)

    def __repr__(self):
        show = self.interface._show_array(self.name, fmt='csv').split('\n')
        return "SciDBArray({0})".format(show[1])

    def contents(self, **kwargs):
        return repr(self) + '\n' + self.interface._scan_array(self.name,
                                                              **kwargs)

    def nonempty(self):
        """
        Return the number of nonempty elements in the array.
        """
        query = "count({0})".format(self.name)
        response = self.interface._execute_query(query, response=True,
                                                 fmt='csv')
        return int(response.lstrip('count').strip())

    def issparse(self):
        """Check whether array is sparse."""
        return (self.nonempty() < self.size)

    def _download_data(self, transfer_bytes=True, output='auto'):
        """Utility routine to download data from SciDB database.
        
        Parameters
        ----------
        transfer_bytes : boolean
            if True (default), then transfer data as bytes rather than as
            ASCII.  This will lead to less approximation of floating point
            data, but for sparse arrays will result in two scan operations,
            one for indices, and one for values.
        output : string
            the output format.  The following are supported:
            
            - 'auto' : choose the best representation for the data
            - 'dense' : return a dense (numpy) array
            - 'sparse' : return a record array containing the indices and
                         values of non-empty elements.
        """
        if output not in ['auto', 'dense', 'sparse']:
            raise ValueError("unrecognized output: '{0}'".format(output))

        dtype = self.datashape.dtype
        sdbtype = self.sdbtype
        shape = self.datashape.shape
        array_is_sparse = self.issparse()
        full_dtype = self.datashape.ind_attr_dtype

        if transfer_bytes:
            # Get byte-string containing array data.
            # This is needed for both sparse and dense outputs
            bytes_rep = self.interface._scan_array(self.name, n=0,
                                                   fmt=self.sdbtype.bytes_fmt)
            bytes_arr = np.fromstring(bytes_rep, dtype=dtype)

        if array_is_sparse:
            # perform a CSV query to find all non-empty index tuples.
            str_rep = self.interface._scan_array(self.name, n=0, fmt='csv+')
            fhandle = cStringIO.StringIO(str_rep)

            # sanity check: make sure our labels are correct
            if list(full_dtype.names) != fhandle.readline().strip().split(','):
                print full_dtype.names
                fhandle.reset()
                print fhandle.readline().strip().split(',')
                raise ValueError("labels are off...")
            fhandle.reset()

            # convert the ASCII representation into a numpy record array
            arr = np.genfromtxt(fhandle, delimiter=',', skip_header=1,
                                dtype=full_dtype)
            
            if transfer_bytes:
                # replace parsed ASCII columns with more accurate bytes
                names = self.sdbtype.names
                if len(names) == 1:
                    arr[names[0]] = bytes_arr
                else:
                    for name in self.sdbtype.names:
                        arr[name] = bytes_arr[name]

            if output == 'dense':
                from scipy import sparse
                data = arr[full_dtype.names[2]]
                ij = (arr[full_dtype.names[0]], arr[full_dtype.names[1]])
                arr_coo = sparse.coo_matrix((data, ij), shape=self.shape)
                arr = arr_coo.toarray()
                
        else:
            if transfer_bytes:
                # reshape bytes_array to the correct shape
                arr = bytes_arr.reshape(shape)
            else:
                # transfer via ASCII.  Here we don't need the indices, so we
                # use 'csv' output.
                str_rep = self.interface._scan_array(self.name, n=0, fmt='csv')
                fhandle = cStringIO.StringIO(str_rep)
                arr = np.genfromtxt(fhandle, delimiter=',', skip_header=1,
                                    dtype=dtype).reshape(shape)

            if output == 'sparse':
                index_arrays = map(np.ravel,
                                   np.meshgrid(*[np.arange(s)
                                                 for s in self.shape],
                                               indexing='ij'))
                arr = arr.ravel()
                if len(sdbtype.names) == 1:
                    value_arrays = [arr]
                else:
                    value_arrays = [arr[col] for col in sdbtype.names]
                arr = np.rec.fromarrays(index_arrays + value_arrays,
                                        dtype=full_dtype)

        return arr

    def toarray(self, transfer_bytes=True):
        """Download data from the server and store in an array.

        Parameters
        ----------
        transfer_bytes : boolean
            if True (default), then transfer data as bytes rather than as
            ASCII.

        Returns
        -------
        arr : np.ndarray
            The dense array containing the data.
        """
        return self._download_data(transfer_bytes=transfer_bytes,
                                   output='dense')

    def tosparse(self, sparse_fmt='recarray', transfer_bytes=True):
        """Download data from the server and store in a sparse array.

        Parameters
        ----------
        transfer_bytes : boolean
            if True (default), then transfer data as bytes rather than as
            ASCII.  This is more accurate, but requires two passes over
            the data (one for indices, one for values).
        sparse_format : string or None
            Specify the sparse format to use.  Available formats are:
            - 'recarray' : a record array containing the indices and
              values for each data point.  This is valid for arrays of
              any dimension and with any number of attributes.
            - ['coo'|'csc'|'csr'|'dok'|'lil'] : a scipy sparse matrix.
              These are valid only for 2-dimensional arrays with a single
              attribute.

        Returns
        -------
        arr : ndarray or sparse matrix
            The sparse representation of the data
        """
        if sparse_fmt == 'recarray':
            spmat = None
        else:
            from scipy import sparse
            try:
                spmat = getattr(sparse, sparse_fmt + "_matrix")
            except AttributeError:
                raise ValueError("Invalid matrix format: "
                                 "'{0}'".format(sparse_fmt))

        columns = self._download_data(transfer_bytes=transfer_bytes,
                                      output='sparse')

        if sparse_fmt == 'recarray':
            return columns
        else:
            from scipy import sparse
            full_dtype = self.datashape.ind_attr_dtype
            if self.ndim != 2:
                raise ValueError("Only recarray format is valid for arrays "
                                 "with ndim != 2.")
            if len(full_dtype.names) > 3:
                raise ValueError("Only recarray format is valid for arrays "
                                 "with multiple attributes.")

            labels = full_dtype.names
            data = columns[labels[2]]
            ij = (columns[labels[0]], columns[labels[1]])
            arr = sparse.coo_matrix((data, ij), shape=self.shape)
            return spmat(arr)

    def __getitem__(self, slices):
        # TODO: handle integer or None indices
        # TODO: use slice() to enable slicing
        # TODO: check for __len__

        # Note that slice steps must be a divisor of the chunk size.
        if len(slices) < self.ndim:
            slices = list(slices) + [slice(None)
                                     for i in range(self.ndim - len(slices))]
        if len(slices) != self.ndim:
            raise ValueError("too many indices")

        indices = [sl.indices(sh) for sl, sh in zip(slices, self.shape)]

        # TODO: do this more efficiently:
        #       check if subarray/thin is needed?
        #       do this in one step, without tmp array?
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
        # TODO: allow broadcasting operations through the use of cross-join.
        # TODO: if other and self point to the same array, this breaks.
        if isinstance(other, SciDBArray):
            if self.shape != other.shape:
                raise NotImplementedError("array shapes must match")
            attr = _new_attribute_label('x', self, other)
            op = op.format(left='{A.a0f}', right='{B.a0f}')
            query = ("store(project(apply(join({A},{B}),{attr},"
                     + op + "), {attr}), {arr})")
        elif np.isscalar(other):
            attr = _new_attribute_label('x', self)
            op = op.format(left='{A.a0f}', right='{B}')
            query = ("store(project(apply({A},{attr},"
                     + op + "),{attr}),{arr})")
        else:
            raise ValueError("unrecognized value: {0}".format(other))

        arr = self.interface.new_array()
        self.interface.query(query,A=self, B=other, attr=attr, arr=arr)
        return arr

    def __add__(self, other):
        return self._join_operation(other, '{left}+{right}')

    def __sub__(self, other):
        return self._join_operation(other, '{left}-{right}')

    def __mul__(self, other):
        return self._join_operation(other, '{left}*{right}')

    def __div__(self, other):
        return self._join_operation(other, '{left}/{right}')

    def __mod__(self, other):
        return self._join_operation(other, '{left}%{right}')

    def __pow__(self, other):
        return self._join_operation(other, 'pow({left},{right})')

    def transpose(self):
        arr = self.interface.new_array()
        self.interface.query("store(transpose({0}), {1})",
                             self, arr)
        return arr

    T = property(transpose)
