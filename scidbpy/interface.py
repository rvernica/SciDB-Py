"""
Low-level interface to Scidb
"""
import abc
import urllib2
from .scidbarray import SciDBArray, SciDBDataShape, SciDBAttribute
from .errors import SHIM_ERROR_DICT

SCIDB_RAND_MAX = 2147483647  # 2 ** 31 - 1


class SciDBInterface(object):
    """SciDBInterface Abstract Base Class.

    This class provides a wrapper to the low-level interface to sciDB.  The
    actual communication with the database should be implemented in
    subclasses
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _execute_query(self, query, response=False, n=0, fmt='auto'):
        """Execute a query on the SciDB engine"""
        pass

    @abc.abstractmethod
    def _upload_bytes(self, data):
        """Upload binary data to the SciDB engine"""
        pass

    def _db_array_name(self):
        # TODO: perhaps use a unique hash for each python session?
        #       Otherwise two sessions connected to the same database
        #       will likely overwrite each other or result in errors.
        if not hasattr(self, 'array_count'):
            self.array_count = 1
        else:
            self.array_count += 1
        return "pyarray%.4i" % self.array_count

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
            query = "CREATE ARRAY {0} {1}".format(name, datashape.descr)
            self._execute_query(query)
        else:
            datashape = None
        return SciDBArray(datashape, self, name, persistent=persistent)

    def _format_query_string(self, query, *args, **kwargs):
        """Format query string.

        See query() documentation for more information
        """
        parse = SciDBAttribute.parse
        args = (parse(v) for v in args)
        kwargs = dict((k, parse(v)) for k, v in kwargs.iteritems())
        query = query.format(*args, **kwargs)
        return query

    def query(self, query, *args, **kwargs):
        """Perform a query on the database.

        This wraps a query constructor which allows the creation of
        sophisticated SciDB queries which act on arrays wrapped by SciDBArray
        objects.  See below for details.

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
        >>> arr = sdb.new_array((3, 3), dtype, **kwargs)
        >>> sdb.query('store(build({0}, iif({i}={j}, 1, 0)), {0})',
        ...           arr, i=arr.index(0), j=arr.index(1))
        >>> arr.toarray()
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])

        In the query string, substitutions are identified by braces
        ('{' and '}'). The contents of the braces should refer to either an
        argument index (e.g. ``{0}`` references the first non-keyword
        argument) or the name of a keyword argument (e.g. ``{i}`` references
        the keyword argument ``i``).  Arguments of type SciDBAttribute (such
        as SciDBArrays, SciDBArray indices, etc.) have their ``name`` attribute
        inserted.  All other argument types are inserted as-is.
        """
        return self._execute_query(self._format_query_string(query,
                                                             *args, **kwargs))
        
    def list_arrays(self, **kwargs):
        # TODO: parse to a dictionary of names and schemas
        if 'response' not in kwargs:
            kwargs['response'] = True
        return self._execute_query("list('arrays')", **kwargs)

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

    def identity(self, n, dtype='double', **kwargs):
        """Return a 2-dimensional square identity matrix of size n

        Parameters
        ----------
        n : integer
            the number of rows and columns in the matrix
        dtype: string or list
            The data type of the array
        **kwargs:
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray containint an [n x n] identity matrix
        """
        arr = self.new_array((n, n), dtype, **kwargs)
        self.query('store(build({0},iif({i}={j},1,0)),{0})',
                   arr, i=arr.index(0, full=False),
                   j=arr.index(1, full=False))
        return arr

    def dot(self, A, B):
        """Compute the matrix product of A and B"""
        # TODO: implement vector-vector and matrix-vector dot()
        if (A.ndim != 2) or (B.ndim != 2):
            raise ValueError("dot requires 2-dimensional arrays")
        if A.shape[1] != B.shape[0]:
            raise ValueError("array dimensions must match for matrix product")

        result = self.new_array()
        self.query('store(multiply({0},{1}),{2})', A, B, result)
        return result

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

    def from_file(self, filename, **kwargs):
        # TODO: allow creation of arrays from uploaded files
        # TODO: allow creation of arrays from pre-existing files within the
        #       database
        raise NotImplementedError()

    def _apply_func(self, A, func):
        # TODO: new value name could conflict.  How to generate a unique one?
        arr = self.new_array()
        self.query("store(project(apply({0},{1}_val,{1}({2}))"
                   ",{1}_val),{3})", A, func, A.val(0), arr)
        return arr

    def sin(self, A):
        return self._apply_func(A, 'sin')

    def cos(self, A):
        return self._apply_func(A, 'cos')

    def tan(self, A):
        return self._apply_func(A, 'tan')

    def asin(self, A):
        return self._apply_func(A, 'asin')

    def acos(self, A):
        return self._apply_func(A, 'acos')

    def atan(self, A):
        return self._apply_func(A, 'atan')

    def exp(self, A):
        return self._apply_func(A, 'exp')

    def log(self, A):
        return self._apply_func(A, 'log')

    def log10(self, A):
        return self._apply_func(A, 'log10')

    def _aggregate(self, A, agg, ind=None):
        # TODO: new value name could conflict.  How to generate a unique one?
        # TODO: ind behavior does not match numpy.  How to proceed?

        if ind is None:
            qstring = "store(aggregate({0}, {agg}({val})), {1})"
            index = ''
        else:
            try:
                ind = tuple(ind)
            except:
                ind = (ind,)

            index_fmt = ', '.join(['{%i}' % i for i in range(len(ind))])
            index = self._format_query_string(index_fmt, *[A.index(i) for
                                                           i in ind])

            qstring = ("store(aggregate({0}, {agg}({val}), {index}), {1})")
        
        arr = self.new_array()
        self.query(qstring, A, arr, val=A.val(0), agg=agg, index=index)
        return arr

    def min(self, A, index=None):
        return self._aggregate(A, 'min', index)

    def max(self, A, index=None):
        return self._aggregate(A, 'max', index)

    def sum(self, A, index=None):
        return self._aggregate(A, 'sum', index)

    def var(self, A, index=None):
        return self._aggregate(A, 'var', index)

    def stdev(self, A, index=None):
        return self._aggregate(A, 'stdev', index)

    def std(self, A, index=None):
        return self._aggregate(A, 'stdev', index)

    def avg(self, A, index=None):
        return self._aggregate(A, 'avg', index)

    def mean(self, A, index=None):
        return self._aggregate(A, 'avg', index)

    def count(self, A, index=None):
        return self._aggregate(A, 'count', index)

    def approxdc(self, A, index=None):
        return self._aggregate(A, 'approxdc', index)

    def pairwise_distances(self, A, B):
        tmp = self.new_array()
        self.query(("store(project(apply(cross_join({A}, {B}, {Aj}, {Bj}),"
                    "                    sqdiff,"
                    "                    pow({Aval} - {Bval}, 2)),"
                    "              sqdiff),"
                    "      {tmp})"), A=A, B=B, Aj=A.index(1), Bj=B.index(1),
                   Aval=A.val(0), Bval=B.val(0), tmp=tmp)
        result = self.new_array()
        self.query(("store(project(apply(sum({tmp}, {val}, {i0}, {i2}),"
                    "                    {val}, sqrt({val}_sum)), {val}),"
                    "      {result})"), tmp=tmp, val=tmp.val(0, full=False),
                   i0=tmp.index(0), i2=tmp.index(2), result=result)
        return result


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
            url += "&save={0}".format(save)

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
