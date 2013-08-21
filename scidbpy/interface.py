"""
Low-level Interface to SciDB
============================
These interfaces are designed to be extensible and allow various interfaces
to the SciDB engine.

The following interfaces are currently available:

- SciDBShimInterface : interface via HTTP using Shim [1]_

[1] https://github.com/Paradigm4/shim
"""
# License: Simplified BSD, 2013
# Author: Jake Vanderplas <jakevdp@cs.washington.edu>

import abc
import urllib2
import re
import csv
import numpy as np
from .scidbarray import SciDBArray, SciDBDataShape, ArrayAlias
from .errors import SHIM_ERROR_DICT
from .utils import broadcastable

__all__ = ['SciDBInterface', 'SciDBShimInterface']

SCIDB_RAND_MAX = 2147483647  # 2 ** 31 - 1


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


class SciDBInterface(object):
    """SciDBInterface Abstract Base Class.

    This class provides a wrapper to the low-level interface to sciDB.  The
    actual communication with the database should be implemented in
    subclasses.

    Subclasses should implement the following methods, with descriptions given
    below:

    - ``_execute_query``
    - ``_upload_bytes``
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _execute_query(self, query, response=False, n=0, fmt='auto'):
        """Execute a query on the SciDB engine

        Parameters
        ----------
        query : string
            A string representing a SciDB query.  This string will be
            executed on the engine.
        response : boolean (optional)
            Indicate whether the query returns a response.  If True, then
            the response is formatted according to the ``n`` and ``fmt``
            keywords.  Default is False.
        n : integer
            number of lines to return.  If n=0 (default), then return all
            lines.  Only accessed if response=True.
        fmt : str
            format code for the returned values.  Options are:
            =================== ===============================================
            Format Code         Description
            =================== ===============================================
            auto (default)      SciDB array format.
            csv	                Comma-separated values (CSV)
            csv+                CSV with dimension indices.
            dcsv                CSV, distinguishing between dims and attrs
            dense               Ideal for viewing 2-dimensional matrices.
            lcsv+               Like csv+, with a flag indicating empty cells.
            sparse              Sparse SciDB array format.
            lsparse             Sparse format with flag indicating empty cells.
            (type1, type2...)   Raw binary format
            =================== ===============================================
        Returns
        -------
        results : string, bytes, or None
            If response is False, None is returned.  If response is True, then
            either a string or a byte string is returned, depending on the
            value of ``fmt``.
        """
        if not hasattr(self, '_query_log'):
            self._query_log = []
        self._query_log.append(query)

    @abc.abstractmethod
    def _upload_bytes(self, data):
        """Upload binary data to the SciDB engine

        Parameters
        ----------
        data : bytestring
            The raw byte data to upload to a file on the SciDB server.
        Returns
        -------
        filename : string
            The name of the resulting file on the SciDB server.
        """
        pass

    @abc.abstractmethod
    def _get_uid(self):
        """Get a unique query ID from the database"""
        pass

    def _db_array_name(self):
        """Return a unique array name for a new array on the database"""
        arr_key = 'py'

        if not hasattr(self, 'uid'):
            self.uid = self._get_uid()

        if not hasattr(self, 'array_count'):
            self.array_count = 1
        else:
            # on subsequent calls, increment the array count
            self.array_count += 1
        return "{0}{1}_{2:05}".format(arr_key, self.uid, self.array_count)

    def _scan_array(self, name, **kwargs):
        """Return the contents of the given array"""
        if 'response' not in kwargs:
            kwargs['response'] = True
        return self._execute_query("scan({0})".format(name), **kwargs)

    def _show_array(self, name, **kwargs):
        """Show the schema of the given array"""
        if 'response' not in kwargs:
            kwargs['response'] = True
        return self._execute_query("show({0})".format(name), **kwargs)

    def _array_dimensions(self, name, **kwargs):
        """Show the dimensions of the given array"""
        if 'response' not in kwargs:
            kwargs['response'] = True
        return self._execute_query("dimensions({0})".format(name), **kwargs)

    def wrap_array(self, scidbname):
        """
        Create a new SciDBArray object that references an existing SciDB
        array

        Parameters
        ----------
        scidbname : string
            Wrap an existing scidb array referred to by `scidbname`. The
            SciDB array object persistent value will be set to True, and
            the object shape, datashape and data type values will be
            determined by the SciDB array.
        """
        schema = self._show_array(scidbname, fmt='csv')
        datashape = SciDBDataShape.from_schema(schema)
        return SciDBArray(datashape, self, scidbname, persistent=True)

    def new_array(self, shape=None, dtype='double', persistent=False,
                  **kwargs):
        """
        Create a new array, either instantiating it in SciDB or simply
        reserving the name for use in a later query.

        Parameters
        ----------
        shape : int or tuple (optional)
            The shape of the array to create.  If not specified, no array
            will be created and a name will simply be reserved for later use.
            WARNING: if shape=None and persistent=False, an error will result
            when the array goes out of scope, unless the name is used to
            create an array on the server.
        dtype : string (optional)
            the datatype of the array.  This is only referenced if `shape`
            is specified.  Default is 'double'.
        persistent : boolean (optional)
            whether the created array should be persistent, i.e. survive
            in SciDB past when the object wrapper goes out of scope.  Default
            is False.
        **kwargs : (optional)
            If `shape` is specified, additional keyword arguments are passed
            to SciDBDataShape.  Otherwise, these will not be referenced.
        Returns
        -------
        arr : SciDBArray
            wrapper of the new SciDB array instance.
        """
        name = self._db_array_name()
        if shape is not None:
            datashape = SciDBDataShape(shape, dtype, **kwargs)
            query = "CREATE ARRAY {0} {1}".format(name, datashape.schema)
            self._execute_query(query)
        else:
            datashape = None
        return SciDBArray(datashape, self, name, persistent=persistent)

    def _format_query_string(self, query, *args, **kwargs):
        """Format query string.

        See query() documentation for more information
        """
        parse = lambda x: ArrayAlias(x) if isinstance(x, SciDBArray) else x
        args = (parse(v) for v in args)
        kwargs = dict((k, parse(v)) for k, v in kwargs.iteritems())
        query = query.format(*args, **kwargs)
        return query

    def query(self, query, *args, **kwargs):
        """Perform a query on the database.

        This wraps a query constructor which allows the creation of
        sophisticated SciDB queries which act on arrays wrapped by SciDBArray
        objects.  See Notes below for details.

        Parameters
        ----------
        query: string
            The query string, with curly-braces to indicate insertions
        *args, **kwargs:
            Values to be inserted (see below).

        Details
        -------
        The query string uses the python string formatting convention, with
        appropriate substitutions made for arguments or keyword arguments that
        are subclasses of SciDBAttribute, such as SciDBArrays or their indices.

        For example, a 3x3 identity matrix can be created using the query
        function as follows:

        >>> sdb = SciDBShimInterface('http://localhost:8080')
        >>> A = sdb.new_array((3, 3), dtype, **kwargs)
        >>> sdb.query('store(build({A}, iif({A.d0}={A.d1}, 1, 0)), {A})', A=A)
        >>> A.toarray()
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])

        The query string follows the python string formatting conventions,
        where variables to be replaced are indicated by curly braces
        ('{' and '}'). The contents of the braces should refer to either an
        argument index (e.g. ``{0}`` references the first non-keyword
        argument) or the name of a keyword argument (e.g. ``{A}`` references
        the keyword argument ``A``).  Arguments of type SciDBAttribute (such
        as SciDBArrays, SciDBArray indices, etc.) have their ``name`` attribute
        inserted.  Dimension and Attribute names can be inserted using the
        following syntax.  As an example, we'll use an array with the following
        schema:

            myarray<val:double,label:int64> [i=0:4,5,0,j=0:4,5,0]

        The dimension/attribute shortcuts are as follows:

        - A.d0, A.d1, A.d2... : insert array dimension names.
          For the above case, '{A.d0}' will be translated to 'i',
          and '{A.d1}' will be translated to 'j'.
        - A.d0f, A.d1f, A.d2f... : insert fully-qualified dimension names.
          For the above case, '{A.d0f}' will be translated to 'myarray.i',
          and '{A.d1f}' will be translated to 'myarray.j'.
        - A.a0, A.a1, A.a2... : insert array attribute names.
          For the above case, '{A.a0}' will be translated to 'val',
          and '{A.a1}' will be translated to 'label'.
        - A.a0f, A.a1f, A.a2f... : insert fully-qualified attribute names.
          For the above case, '{A.a0f}' will be translated to 'myarray.val',
          and '{A.a1f}' will be translated to 'myarray.label'.

        All other argument types are inserted as-is, i.e. with their string
        representation.
        """
        qstring = self._format_query_string(query, *args, **kwargs)
        return self._execute_query(qstring)

    def list_arrays(self, parsed=True, n=0):
        """List the arrays currently in the database

        Parameters
        ----------
        parsed : boolean
            If True (default), then parse the results into a dictionary of
            array names as keys, schema as values
        n : integer
            the maximum number of arrays to list.  If n=0, then list all

        Returns
        -------
        array_list : string or dictionary
            The list of arrays.  If parsed=True, then the result is returned
            as a dictionary.
        """
        # TODO: more stable way to do this than string parsing?
        arr_list = self._execute_query("list('arrays')", n=n, response=True)
        if parsed:
            if arr_list.strip() == '[]':
                arr_list = {}
            else:
                # find the correct quote char for the output.
                # Depending on the SciDB version, it could be " or '
                quotechar = arr_list[2]
                if quotechar not in ["'", '"']:
                    raise ValueError("output of list('arrays') "
                                     "is not in the expected format")
                R = re.compile(r'\(([^\(\)]*)\)')
                splits = R.findall(arr_list)
                arr_list = dict((a[0], a[1:])
                                for a in csv.reader(splits,
                                                    quotechar=quotechar))
        return arr_list

    def ones(self, shape, dtype='double', **kwargs):
        """Return an array of ones

        Parameters
        ----------
        shape: tuple or int
            The shape of the array
        dtype: string or list
            The data type of the array
        **kwargs:
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray consisting of all ones.
        """
        arr = self.new_array(shape, dtype, **kwargs)
        self.query('store(build({0},1),{0})', arr)
        return arr

    def zeros(self, shape, dtype='double', **kwargs):
        """Return an array of zeros

        Parameters
        ----------
        shape: tuple or int
            The shape of the array
        dtype: string or list
            The data type of the array
        **kwargs:
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray consisting of all zeros.
        """
        schema = SciDBDataShape(shape, dtype, **kwargs).schema
        arr = self.new_array()
        self.query('store(build({0}, 0), {1})', schema, arr)
        return arr

    def random(self, shape, dtype='double', lower=0, upper=1, **kwargs):
        """Return an array of random floats between lower and upper

        Parameters
        ----------
        shape: tuple or int
            The shape of the array
        dtype: string or list
            The data type of the array
        lower: float
            The lower bound of the random sample (default=0)
        upper: float
            The upper bound of the random sample (default=1)
        **kwargs:
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray consisting of random floating point numbers,
            uniformly distributed between `lower` and `upper`.
        """
        # TODO: can be done more efficiently
        #       if lower is 0 or upper - lower is 1
        schema = SciDBDataShape(shape, dtype, **kwargs).schema
        fill_value = ('random()*{0}/{2}+{1}'.format(upper - lower, lower,
                                                    float(SCIDB_RAND_MAX)))
        arr = self.new_array()
        self.query('store(build({0}, {1}), {2})', schema, fill_value, arr)
        return arr

    def randint(self, shape, dtype='uint32', lower=0, upper=SCIDB_RAND_MAX,
                **kwargs):
        """Return an array of random integers between lower and upper

        Parameters
        ----------
        shape: tuple or int
            The shape of the array
        dtype: string or list
            The data type of the array
        lower: float
            The lower bound of the random sample (default=0)
        upper: float
            The upper bound of the random sample (default=2147483647)
        **kwargs:
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray consisting of random integers, uniformly distributed
            between `lower` and `upper`.
        """
        schema = SciDBDataShape(shape, dtype, **kwargs).schema
        fill_value = 'random() % {0} + {1}'.format(upper - lower, lower)
        arr = self.new_array()
        self.query('store(build({0}, {1}), {2})', schema, fill_value, arr)
        return arr

    def arange(self, start, stop=None, step=1, dtype=None, **kwargs):
        """arange([start,] stop[, step,], dtype=None, **kwargs)

        Return evenly spaced values within a given interval.

        Values are generated within the half-open interval ``[start, stop)``
        (in other words, the interval including `start` but excluding `stop`).
        For integer arguments the behavior is equivalent to the Python
        `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
        but returns an ndarray rather than a list.

        When using a non-integer step, such as 0.1, the results will often not
        be consistent.  It is better to use ``linspace`` for these cases.

        Parameters
        ----------
        start : number, optional
            Start of interval.  The interval includes this value.  The default
            start value is 0.
        stop : number
            End of interval.  The interval does not include this value, except
            in some cases where `step` is not an integer and floating point
            round-off affects the length of `out`.
        step : number, optional
            Spacing between values.  For any output `out`, this is the distance
            between two adjacent values, ``out[i+1] - out[i]``.  The default
            step size is 1.  If `step` is specified, `start` must also be
            given.
        dtype : dtype
            The type of the output array.  If `dtype` is not given, it is
            inferred from the type of the input arguments.
        **kwargs :
            Additional arguments are passed to SciDBDatashape when creating
            the output array.

        Returns
        -------
        arange : SciDBArray
            Array of evenly spaced values.

            For floating point arguments, the length of the result is
            ``ceil((stop - start)/step)``.  Because of floating point overflow,
            this rule may result in the last element of `out` being greater
            than `stop`.
        """
        if stop is None:
            stop = start
            start = 0

        if dtype is None:
            dtype = np.array(start + stop + step).dtype

        size = int(np.ceil((stop - start) * 1. / step))

        arr = self.new_array(size, dtype, **kwargs)
        self.query("store(build({A}, {start} + {step} * {A.d0}), {A})",
                   A=arr, start=start, step=step)
        return arr

    def linspace(self, start, stop, num=50,
                 endpoint=True, retstep=False, **kwargs):
        """
        Return evenly spaced numbers over a specified interval.

        Returns `num` evenly spaced samples, calculated over the
        interval [`start`, `stop` ].

        The endpoint of the interval can optionally be excluded.

        Parameters
        ----------
        start : scalar
            The starting value of the sequence.
        stop : scalar
            The end value of the sequence, unless `endpoint` is set to False.
            In that case, the sequence consists of all but the last of
            ``num + 1`` evenly spaced samples, so that `stop` is excluded.
            Note that the step size changes when `endpoint` is False.
        num : int, optional
            Number of samples to generate. Default is 50.
        endpoint : bool, optional
            If True, `stop` is the last sample. Otherwise, it is not included.
            Default is True.
        retstep : bool, optional
            If True, return (`samples`, `step`), where `step` is the spacing
            between samples.
        **kwargs :
            additional keyword arguments are passed to SciDBDataShape

        Returns
        -------
        samples : SciDBArray
            There are `num` equally spaced samples in the closed interval
            ``[start, stop]`` or the half-open interval ``[start, stop)``
            (depending on whether `endpoint` is True or False).
        step : float (only if `retstep` is True)
            Size of spacing between samples.
        """
        num = int(num)

        if endpoint:
            step = (stop - start) * 1. / (num - 1)
        else:
            step = (stop - start) * 1. / num

        arr = self.new_array(num, **kwargs)
        self.query("store(build({A}, {start} + {step} * {A.d0}), {A})",
                   A=arr, start=start, step=step)

        if retstep:
            return arr, step
        else:
            return arr

    def identity(self, n, dtype='double', sparse=False, **kwargs):
        """Return a 2-dimensional square identity matrix of size n

        Parameters
        ----------
        n : integer
            the number of rows and columns in the matrix
        dtype: string or list
            The data type of the array
        sparse: boolean
            specify whether to create a sparse array (default=False)
        **kwargs:
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray containint an [n x n] identity matrix
        """
        arr = self.new_array((n, n), dtype, **kwargs)
        if sparse:
            query = 'store(build_sparse({A},1,{A.d0}={A.d1}),{A})'
        else:
            query = 'store(build({A},iif({A.d0}={A.d1},1,0)),{A})'
        self.query(query, A=arr)
        return arr

    def dot(self, A, B):
        """Compute the matrix product of A and B

        Parameters
        ----------
        A : SciDBArray
            A must be a two-dimensional matrix of shape (n, p)
        B : SciDBArray
            B must be a two-dimensional matrix of shape (p, m)

        Returns
        -------
        C : SciDBArray
            The wrapper of the SciDB Array, of shape (n, m), consisting of the
            matrix product of A and B
        """
        # TODO: use GEMM and repartition where applicable.  GEMM requires the
        #       chunk size to be 32.  We should probably not repartition the
        #       arrays silently, but instead raise an efficiency warning and
        #       provide a flag that enables automatic repartitioning.

        # TODO: make matrix-vector and vector-vector cases more efficient.
        #       Currently they involve creating copies of the arrays, but this
        #       is just a place-holder for a more efficient implementation.

        if A.ndim not in (1, 2) or B.ndim not in (1, 2):
            raise ValueError("dot requires 1 or 2-dimensional arrays")

        if A.shape[-1] != B.shape[0]:
            raise ValueError("array dimensions must match for dot product")

        output_shape = A.shape[:-1] + B.shape[1:]

        # TODO: the following four transformations should be done by building
        #       a single query rather than executing separate queries.
        #       The following should be considered a place-holder for right
        #       now.
        if A.ndim == 1:
            A = A.reshape((1, A.size))

        if B.ndim == 1:
            B = B.reshape((B.size, 1))

        if A.sdbtype.nullable[0]:
            A = A.substitute(0)

        if B.sdbtype.nullable[0]:
            B = B.substitute(0)

        C = self.new_array()
        self.query('store(multiply({0},{1}),{2})', A, B, C)

        if C.shape == output_shape:
            return C
        elif len(output_shape) == 0:
            return C[0, 0]
        else:
            return C.reshape(output_shape)

    def svd(self, A, return_U=True, return_S=True, return_VT=True):
        """Compute the Singular Value Decomposition of the array A:

        A = U.S.V^T

        Parameters
        ----------
        A : SciDBArray
            The array for which the SVD will be computed.  It should be a
            2-dimensional array with a single value per cell.  Currently, the
            svd routine requires non-overlapping chunks of size 32.
        return_U, return_S, return_VT : boolean
            if any is True, then return the associated array.  All are True
            by default

        Returns
        -------
        [U], [S], [VT] : SciDBArrays
            Arrays storing the singular values and vectors of A.
        """
        if (A.ndim != 2):
            raise ValueError("svd requires 2-dimensional arrays")
        self.query("load_library('dense_linear_algebra')")

        out_dict = dict(U=return_U, S=return_S, VT=return_VT)

        # TODO: check that data type is double and chunk size is 32
        ret = []
        for output in ['U', 'S', 'VT']:
            if out_dict[output]:
                ret.append(self.new_array())
                self.query("store(gesvd({0}, '{1}'),{2})", A, output, ret[-1])
        return tuple(ret)

    def from_array(self, A, **kwargs):
        """Initialize a scidb array from a numpy array"""
        # TODO: make this work for other data types
        if A.dtype != 'double':
            raise NotImplementedError("from_array only implemented for double")
        dtype = 'double'
        data = A.tostring(order='C')
        filename = self._upload_bytes(A.tostring(order='C'))
        arr = self.new_array(A.shape, 'double', **kwargs)
        self.query("load({0},'{1}',{2},'{3}')", arr, filename, 0, '(double)')
        return arr

    def toarray(self, A):
        """Convert a SciDB array to a numpy array"""
        return A.toarray()

    def tosparse(self, A, sparse_fmt='recarray', transfer_bytes=True):
        """Convert a SciDB array to a sparse representation"""
        return A.tosparse(sparse_fmt=sparse_fmt, transfer_bytes=transfer_bytes)

    def from_file(self, filename, **kwargs):
        # TODO: allow creation of arrays from uploaded files
        # TODO: allow creation of arrays from pre-existing files within the
        #       database
        raise NotImplementedError()

    def _apply_func(self, A, func):
        # TODO: new value name could conflict.  How to generate a unique one?
        arr = self.new_array()
        self.query("store(project(apply({A},{func}_{A.a0},{func}({A.a0})),"
                   "{func}_{A.a0}), {arr})",
                   A=A, func=func, arr=arr)
        return arr

    def sin(self, A):
        """Element-wise trigonometric sine"""
        return self._apply_func(A, 'sin')

    def cos(self, A):
        """Element-wise trigonometric cosine"""
        return self._apply_func(A, 'cos')

    def tan(self, A):
        """Element-wise trigonometric tangent"""
        return self._apply_func(A, 'tan')

    def asin(self, A):
        """Element-wise trigonometric inverse sine"""
        return self._apply_func(A, 'asin')

    def acos(self, A):
        """Element-wise trigonometric inverse cosine"""
        return self._apply_func(A, 'acos')

    def atan(self, A):
        """Element-wise trigonometric inverse tangent"""
        return self._apply_func(A, 'atan')

    def exp(self, A):
        """Element-wise natural exponent"""
        return self._apply_func(A, 'exp')

    def log(self, A):
        """Element-wise natural logarithm"""
        return self._apply_func(A, 'log')

    def log10(self, A):
        """Element-wise base-10 logarithm"""
        return self._apply_func(A, 'log10')

    def min(self, A, index=None, scidb_syntax=False):
        return A.min(index, scidb_syntax)

    def max(self, A, index=None, scidb_syntax=False):
        return A.max(index, scidb_syntax)

    def sum(self, A, index=None, scidb_syntax=False):
        return A.sum(index, scidb_syntax)

    def var(self, A, index=None, scidb_syntax=False):
        return A.var(index, scidb_syntax)

    def stdev(self, A, index=None, scidb_syntax=False):
        return A.stdev(index, scidb_syntax)

    def std(self, A, index=None, scidb_syntax=False):
        return A.std(index, scidb_syntax)

    def avg(self, A, index=None, scidb_syntax=False):
        return A.avg(index, scidb_syntax)

    def mean(self, A, index=None, scidb_syntax=False):
        return A.mean(index, scidb_syntax)

    def count(self, A, index=None, scidb_syntax=False):
        return A.count(index, scidb_syntax)

    def approxdc(self, A, index=None, scidb_syntax=False):
        return A.approxdc(index, scidb_syntax)

    def substitute(self, A, value):
        return A.substitute(value)

    def join(self, *args):
        """
        Perform a series of array joins on the arguments
        and return the result.
        """
        last = args[0]
        for arg in args[1:]:
            arr = self.new_array()
            self.query('store(join({0},{1}), {2})', last, arg, arr)
            last = arr
        return last

    def cross_join(self, A, B, *dims):
        """Perform a cross-join on arrays A and B.

        Parameters
        ----------
        A, B : SciDBArray
        *dims : tuples
            The remaining arguments are tuples of dimension indices which
            should be joined.
        """
        dims = ','.join(" {{A.d{0}f}}, {{B.d{1}f}}".format(*tup)
                        for tup in dims)
        if dims:
            dims = ',' + dims
        query = ('store(cross_join({A}, {B}'
                 + dims
                 + '), {arr})')
        arr = self.new_array()
        self.query(query, A=A, B=B, arr=arr)
        return arr

    def _join_operation(self, left, right, op):
        """Perform a join operation across arrays or values.

        See e.g. SciDBArray.__add__ for an example usage.
        """
        if isinstance(left, SciDBArray):
            left_name = left.name
            left_fmt = '{left.a0f}'
            left_is_sdb = True
        else:
            left_name = None
            left_fmt = '{left}'
            left_is_sdb = False

        if isinstance(right, SciDBArray):
            right_name = right.name
            right_fmt = '{right.a0f}'
            right_is_sdb = True
        else:
            right_name = None
            right_fmt = '{right}'
            right_is_sdb = False

        # some common names needed below
        op = op.format(left=left_fmt, right=right_fmt)
        aL = aR = None
        permutation = None

        # Neither entry is a SciDBArray
        if not (left_is_sdb or right_is_sdb):
            raise ValueError("One of left/right needs to be a SciDBArray")

        # Both entries are SciDBArrays
        elif (left_is_sdb and right_is_sdb):
            # array shapes match: use a join
            if left.shape == right.shape:
                if (left.chunk_size != right.chunk_size or
                    left.chunk_overlap != right.chunk_overlap):
                    raise ValueError("join operations require chunk_size/"
                                     "chunk_overlap to match.")
                attr = _new_attribute_label('x', left, right)
                if left_name == right_name:
                    # same array: we can do this without a join
                    query = ("store(project(apply({left}, {attr}, "
                             + op + "), {attr}), {arr})")
                else:
                    query = ("store(project(apply(join({left},{right}),{attr},"
                             + op + "), {attr}), {arr})")

            # array shapes are broadcastable: use a cross_join
            elif broadcastable(left.shape, right.shape):
                join_indices = []
                left_slices = []
                right_slices = []

                for tup in zip(reversed(list(enumerate(left.shape))),
                               reversed(list(enumerate(right.shape)))):
                    (i1, s1), (i2, s2) = tup
                    if (s1 == s2):
                        join_indices.append((i1, i2))
                        if (left.chunk_size[i1] != right.chunk_size[i2] or
                            left.chunk_overlap[i1] != right.chunk_overlap[i2]):
                            raise ValueError("join operations require chunk_"
                                             "size/chunk_overlap to match.")
                    elif s1 == 1:
                        left_slices.append(i1)
                    elif s2 == 1:
                        right_slices.append(i2)
                    else:
                        # should never get here, but just in case...
                        raise ValueError("shapes cannot be broadcast")

                # build the left slice query if needed
                if left_slices:
                    dims = ','.join("{{left.d{0}f}},0".format(sl)
                                    for sl in left_slices)
                    left_query = "slice({left}," + dims + ") as {aL}"
                    aL = ArrayAlias(left, "alias_left")
                else:
                    left_query = "{aL}"
                    aL = left

                # build the right slice query if needed
                if right_slices:
                    dims = ','.join("{{right.d{0}f}},0".format(sl)
                                    for sl in right_slices)
                    right_query = "slice({right}," + dims + ") as {aR}"
                    aR = ArrayAlias(right, "alias_right")
                else:
                    right_query = "{aR}"
                    aR = right

                # build the cross_join query
                dims = ','.join("{{aL.d{0}f}}, {{aR.d{1}f}}".format(i, j)
                                for i, j in join_indices)
                if dims:
                    dims = ',' + dims

                query = ("store(project(apply(cross_join(" +
                         left_query + "," + right_query + dims + "),{attr}," +
                         op + "), {attr}), {arr})")
                attr = _new_attribute_label('x', left, right)

                # determine the dimension permutation
                # Here's the problem: cross_join puts all the left array dims
                # first, and the right array dims second.  This is different
                # than numpy's broadcast behavior.  It's also difficult to do
                # in a single operation because conflicting dimension names
                # have a rename scheme that might be difficult to duplicate.
                # So we compromise, and perform a dimension permutation on
                # the result if needed.
                left_shape = list(left.shape)
                right_shape = list(right.shape)
                i_left = 0
                i_right = len(join_indices) + len(right_slices)
                permutation = [-1] * max(left.ndim, right.ndim)

                # first pad the shapes so they're the same length
                if left.ndim > right.ndim:
                    i_right += left.ndim - right.ndim
                    right_shape = [-1] * (left.ndim - right.ndim) + right_shape
                else:
                    left_shape = [-1] * (right.ndim - left.ndim) + left_shape

                # now loop through dimensions and build permutation
                for i, (L, R) in enumerate(zip(left_shape, right_shape)):
                    if L == R or R == -1 or (R == 1 and L >= 0):
                        permutation[i] = i_left
                        i_left += 1
                    elif L == -1 or (L == 1 and R >= 0):
                        permutation[i] = i_right
                        i_right += 1
                    else:
                        # This should never happen, but just to be sure...
                        raise ValueError("shapes are not compatible")

                if permutation == range(len(permutation)):
                    permutation = None

            else:
                raise ValueError("Array of shape {0} can not be "
                                 "broadcasted with array of shape "
                                 "{1}".format(left.shape, right.shape))

        # only left entry is a SciDBArray
        elif left_is_sdb:
            try:
                _ = float(right)
            except:
                raise ValueError("rhs must be a scalar or SciDBArray")
            attr = _new_attribute_label('x', left)
            query = ("store(project(apply({left}, {attr}, "
                     + op + "), {attr}), {arr})")

        # only right entry is a SciDBArray
        elif right_is_sdb:
            try:
                _ = float(left)
            except:
                raise ValueError("lhs must be a scalar or SciDBArray")
            attr = _new_attribute_label('x', right)
            query = ("store(project(apply({right}, {attr}, "
                     + op + "), {attr}), {arr})")

        arr = self.new_array()
        self.query(query, left=left, right=right,
                   aL=aL, aR=aR,
                   attr=attr, arr=arr)

        # reorder the dimensions if needed (for cross_join)
        if permutation is not None:
            arr = arr.transpose(permutation)
        return arr


class SciDBShimInterface(SciDBInterface):
    """HTTP interface to SciDB via shim [1]_

    Parameters
    ----------
    hostname : string
        A URL pointing to a running shim/SciDB session

    [1] https://github.com/Paradigm4/shim
    """
    def __init__(self, hostname):
        self.hostname = hostname.rstrip('/')
        try:
            urllib2.urlopen(self.hostname)
        except urllib2.HTTPError:
            raise ValueError("Invalid hostname: {0}".format(self.hostname))

    def _get_uid(self):
        # load a library to get a query id
        session = self._shim_new_session()
        uid = self._shim_execute_query(session,
                                       "load_library('dense_linear_algebra')",
                                       release=True)
        return uid

    def _execute_query(self, query, response=False, n=0, fmt='auto'):
        # log the query
        SciDBInterface._execute_query(self, query, response, n, fmt)

        session_id = self._shim_new_session()
        if response:
            self._shim_execute_query(session_id, query, save=fmt,
                                     release=False)

            if fmt.startswith('(') and fmt.endswith(')'):
                # binary format
                result = self._shim_read_bytes(session_id, n)
            else:
                # text format
                result = self._shim_read_lines(session_id, n)
            self._shim_release_session(session_id)
        else:
            self._shim_execute_query(session_id, query, release=True)
            result = None
        return result

    def _upload_bytes(self, data):
        session_id = self._shim_new_session()
        return self._shim_upload_file(session_id, data)

    def _shim_url(self, keyword, **kwargs):
        url = self.hostname + '/' + keyword
        if kwargs:
            url += '?' + '&'.join(['{0}={1}'.format(key, val)
                                   for key, val in kwargs.iteritems()])
        return url

    def _shim_urlopen(self, url):
        try:
            return urllib2.urlopen(url)
        except urllib2.HTTPError as e:
            Error = SHIM_ERROR_DICT[e.code]
            raise Error("[HTTP {0}] {1}".format(e.code, e.read()))

    def _shim_new_session(self):
        """Request a new HTTP session from the service"""
        url = self._shim_url('new_session')
        result = self._shim_urlopen(url)
        session_id = int(result.read())
        return session_id

    def _shim_release_session(self, session_id):
        url = self._shim_url('release_session', id=session_id)
        result = self._shim_urlopen(url)

    def _shim_execute_query(self, session_id, query, save=None, release=False):
        url = self._shim_url('execute_query',
                             id=session_id,
                             query=urllib2.quote(query),
                             release=int(bool(release)))
        if save is not None:
            url += "&save={0}".format(urllib2.quote(save))

        # Don't know the accepted way in Python to do something like this.
        # For now, just set the 'debug' attribute on this SciDB array object.
        if hasattr(self, 'debug'):
            print(query)

        result = self._shim_urlopen(url)
        query_id = result.read()
        return query_id

    def _shim_cancel(self, session_id):
        url = self._shim_url('cancel', id=session_id)
        result = self._shim_urlopen(url)

    def _shim_read_lines(self, session_id, n):
        url = self._shim_url('read_lines', id=session_id, n=n)
        result = self._shim_urlopen(url)
        text_result = result.read()
        return text_result

    def _shim_read_bytes(self, session_id, n):
        url = self._shim_url('read_lines', id=session_id, n=n)
        result = self._shim_urlopen(url)
        bytes_result = result.read()
        return bytes_result

    def _shim_upload_file(self, session_id, data):
        # TODO: can this be implemented in urllib2 to remove dependency?
        import requests
        url = self._shim_url('upload_file', id=session_id)
        result = requests.post(url, files=dict(fileupload=data))
        scidb_filename = result.text.strip()
        return scidb_filename
