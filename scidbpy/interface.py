"""
Low-level interface to Scidb
"""
import abc
import urllib2
from .scidbarray import SciDBArray, SciDBDataShape, SciDBAttribute
from .errors import SHIM_ERROR_DICT


class SciDBInterface(object):
    """SciDBInterface Abstract Base Class.

    This class provides a wrapper to the low-level interface to sciDB.  The
    actual communication with the database should be implemented in
    subclasses
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        # Array count will facilitate the creation of unique array names
        # This should be called with super() in subclasses
        self.array_count = 0

    @abc.abstractmethod
    def _execute_query(self, query, response=False, n=0, fmt='auto'):
        """Execute a query on the SciDB engine"""
        pass

    @abc.abstractmethod
    def _upload_bytes(self, data):
        """Upload binary data to the SciDB engine"""
        pass

    def _next_name(self):
        # TODO: perhaps use a unique hash for this session?
        #       Otherwise two python sessions connected to the same database
        #       will likely overwrite each other or result in errors.
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
        name = self._next_name()
        if shape is not None:
            datashape = SciDBDataShape(shape, dtype, **kwargs)
            query = "CREATE ARRAY {0} {1}".format(name, datashape.descr)
            self._execute_query(query)
        else:
            datashape = None
        return SciDBArray(datashape, self, name, persistent=persistent)

    def query(self, query, *args, **kwargs):
        """Perform a query on the database.

        TODO: write some examples
        """
        args = [arg.name for arg in args]
        kwargs = dict([(k, v.name) for k, v in kwargs.iteritems()])
        query = query.format(*args, **kwargs)
        self._execute_query(query)
        
    def list_arrays(self, **kwargs):
        # TODO: return as a dictionary of names and schemas
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
        fill_value = ('random() * {0} / 2147483647.0 + {1}'
                      .format(upper - lower, lower))
        self.query('store(build({0},' + fill_value + '),{0})', arr)
        return arr

    def randint(self, shape, dtype='uint32', lower=0, upper=2147483647, 
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
        self.query('store(build({0},' + fill_value + '),{0})', arr)
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
                   arr, i=arr.index(0), j=arr.index(1))
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

        argdict = dict(U=return_U, S=return_S, VT=return_VT)

        # TODO: check that data type is double and chunk size is 32
        ret = []
        for arg in ['U', 'S', 'VT']:
            if argdict[arg]:
                ret.append(self.new_array())
                self.query("store(gesvd({0},'{1}'),{2})",
                           A, SciDBAttribute(arg), ret[-1])
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
        self.query("load({0},'{1}',-1,'(double)')",
                   arr, SciDBAttribute(filename))
        return arr

    def toarray(self, A):
        """Convert a SciDB array to a numpy array"""
        return A.toarray()

    def from_file(self, filename, **kwargs):
        # TODO: allow creation of arrays from uploaded files
        # TODO: allow creation of arrays from pre-existing files within the
        #       database
        raise NotImplementedError()


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
        SciDBInterface.__init__(self)

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
