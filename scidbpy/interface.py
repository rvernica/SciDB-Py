"""
Low-level Interface to SciDB
============================
These interfaces are designed to be extensible and allow various interfaces
to the SciDB engine.

The following interfaces are currently available:

- 
"""
# License: Simplified BSD, 2013
# Author: Jake Vanderplas <jakevdp@cs.washington.edu>

import abc
import urllib2
import re
import csv
from .scidbarray import SciDBArray, SciDBDataShape, ArrayAlias
from .errors import SHIM_ERROR_DICT

__all__ = ['SciDBInterface', 'SciDBShimInterface']

SCIDB_RAND_MAX = 2147483647  # 2 ** 31 - 1


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

    def _db_array_name(self):
        """Return a unique array name for a new array on the database"""
        # TODO: perhaps use a unique hash for each session?
        #       Otherwise two sessions connected to the same database
        #       will likely overwrite each other or result in errors.
        arr_key = 'pyarray'

        if not hasattr(self, 'array_count'):
            # for the first number, search database to make sure there are
            # no collisions
            current = self.list_arrays(parsed=False)
            nums = map(int, re.findall("\"{0}(\d+)\"".format(arr_key),
                                       current))
            nums.append(0)
            self.array_count = max(nums) + 1
        else:
            # on subsequent calls, increment the array count
            self.array_count += 1
        return "{0}{1:05}".format(arr_key, self.array_count)

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

    # TODO: allow creation of arrays wrapping pre-existing objects within
    #       the database?
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
        dtype : string (optional)
            the datatype of the array.  This is only referenced if `shape`
            is specified.  Default is 'double'.
        persistent : string (optional)
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
            R = re.compile(r'\(([^\(\)]*)\)')
            splits = R.findall(arr_list)
            arr_list = dict((a[0], a[1:]) for a in csv.reader(splits))
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
        arr = self.new_array(shape, dtype, **kwargs)
        self.query('store(build({0},0),{0})', arr)
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
        arr = self.new_array(shape, dtype, **kwargs)
        fill_value = ('random()*{0}/{2}+{1}'.format(upper - lower, lower,
                                                    float(SCIDB_RAND_MAX)))
        self.query('store(build({0}, {1}), {0})', arr, fill_value)
        return arr

    def randint(self, shape, dtype='uint32',
                lower=0, upper=SCIDB_RAND_MAX, 
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
        arr = self.new_array(shape, dtype, **kwargs)
        fill_value = 'random() % {0} + {1}'.format(upper - lower, lower)
        self.query('store(build({0}, {1}), {0})', arr, fill_value)
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
            query ='store(build_sparse({A},1,{A.d0}={A.d1}),{A})'
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
        # TODO: implement vector-vector and matrix-vector dot()
        if (A.ndim != 2) or (B.ndim != 2):
            raise ValueError("dot requires 2-dimensional arrays")
        if A.shape[1] != B.shape[0]:
            raise ValueError("array dimensions must match for matrix product")

        C = self.new_array()
        self.query('store(multiply({0},{1}),{2})', A, B, C)
        return C

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
        self.query("load({0},'{1}',{2},'{3}')", arr, filename, -1, '(double)')
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

    def min(self, A, index=None):
        return A.min(index)

    def max(self, A, index=None):
        return A.max(index)

    def sum(self, A, index=None):
        return A.sum(index)

    def var(self, A, index=None):
        return A.var(index)

    def stdev(self, A, index=None):
        return A.stdev(index)

    def std(self, A, index=None):
        return A.std(index)

    def avg(self, A, index=None):
        return A.avg(index)

    def mean(self, A, index=None):
        return A.mean(index)

    def count(self, A, index=None):
        return A.count(index)

    def approxdc(self, A, index=None):
        return A.approxdc(index)

    #def pairwise_distances(self, X, Y=None):
    #    """Compute the pairwise distances between arrays X and Y"""
    #    if Y is None:
    #        Y = X
    #
    #    assert X.ndim == 2
    #    assert Y.ndim == 2
    #    assert X.shape[1] == Y.shape[1]
    #
    #    D = self.new_array()
    #    query = ("store("
    #             "  aggregate("
    #             "    project("
    #             "      apply("
    #             "        cross_join({X} as X1,"
    #             "                   {Y} as X2,"
    #             "                   X1.{Xj}, X2.{Yj}),"
    #             "        {d}, (X1.{Xv} - X2.{Yv}) * (X1.{Xv} - X2.{Yv})),"
    #             "      {d}),"
    #             "    sum({d}), X1.{Xi}, X2.{Yi}),"
    #             "  {D})")
    #
    #    self.query(query,
    #               X=X, Xj=X.index(1, full=False),
    #               Xv=X.val(0, full=False),
    #               Xi=X.index(0, full=False),
    #               Y=Y, Yj=Y.index(1, full=False),
    #               Yv=Y.val(0, full=False),
    #               Yi=Y.index(0, full=False),
    #               D=D, d='d')
    #    return D

    def join(self, A, B):
        """Perform a simple array join"""
        arr = self.new_array()
        self.query('store(join({0},{1}), {2})', A, B, arr)
        return arr

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
        query = ('store(cross_join({A}, {B},'
                 + dims
                 + '), {arr})')
        arr = self.new_array()
        self.query(query, A=A, B=B, arr=arr)
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
        except HTTPError:
            raise ValueError("Invalid hostname: {0}".format(self.hostname))

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
