"""
Low-level interface to Scidb
"""
import abc
import urllib2
from .scidbarray import SciDBArray, SciDBDataShape


class SciDBError(Exception):
    pass


class SciDBInvalidQuery(SciDBError):
    pass


class SciDBInvalidSession(SciDBError):
    pass


class SciDBEndOfFile(SciDBError):
    pass


class SciDBInvalidRequest(SciDBError):
    pass


class SciDBQueryError(SciDBError):
    pass


class SciDBConnectionError(SciDBError):
    pass


class SciDBMemoryError(SciDBError):
    pass


class SciDBUnknownError(SciDBError):
    pass


SHIM_ERROR_DICT = {400: SciDBInvalidQuery,
                   404: SciDBInvalidSession,
                   410: SciDBEndOfFile,
                   414: SciDBInvalidRequest,
                   500: SciDBQueryError,
                   503: SciDBConnectionError,
                   507: SciDBMemoryError}


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

    def _create_array(self, desc, name=None, **kwargs):
        """Utility routine to create a new array

        Note that this routine does not build the array: it only creates
        an empty array.

        Parameters
        ----------
        desc : string
            Array descriptor.  See SciDB documentation for details.
        name : string (optional)
            The name of the array to create.  An error will be raised if
            an array with this name already exists in the database.  If
            not specified, a name will be generated.

        Returns
        -------
        name : string
            the name of the stored array

        See Also
        --------
        _build_array : build and fill an array
        """
        if name is None:
            name = self._next_name()
        query = "CREATE ARRAY {0} {1}".format(name, desc)
        self._execute_query(query, **kwargs)
        return name

    def _build_array(self, desc, name=None, fill_value=1, **kwargs):
        """Utility routine to build a new array

        Parameters
        ----------
        desc : string
            Array descriptor.  See SciDB documentation for details.
        name : string (optional)
            The name of the array to create.  An error will be raised if
            an array with this name already exists in the database.  If
            not specified, a name will be generated.
        fill_value : integer, float, or string (optional)
            The value with with the array should be filled.  This may contain
            a string expression referencing the dimension indices. Default = 1.

        Returns
        -------
        name : string
            the name of the stored array

        See Also
        --------
        _create_array : create an empty array
        """
        if name is None:
            name = self._next_name()
        query = "store(build({0},{1}),{2})".format(desc, fill_value, name)
        self._execute_query(query, **kwargs)
        return name

    def _delete_array(self, name, **kwargs):
        """Utility routine to delete an existing array

        Parameters
        ----------
        name : string
            The name of the array to delete.  An error will be raised if
            an array with this name does not exist in the database.
        """
        self._execute_query("remove({0})".format(name), **kwargs)

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

    def list_arrays(self, **kwargs):
        # TODO: return as a dictionary of names and schemas
        if 'response' not in kwargs:
            kwargs['response'] = True
        return self._execute_query("list('arrays')", **kwargs)

    # TODO: allow creation of arrays wrapping persistent memory?

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
        datashape = SciDBDataShape(shape, dtype, **kwargs)
        name = self._build_array(datashape.descr, fill_value=1)
        return SciDBArray(datashape, self, name)

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
        datashape = SciDBDataShape(shape, dtype, **kwargs)
        name = self._build_array(datashape.descr, fill_value=0)
        return SciDBArray(datashape, self, name)

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
        datashape = SciDBDataShape(shape, dtype, **kwargs)
        fill_value = 'random() * {0} / 2147483647.0 + {1}'.format(
            upper - lower, upper)
        name = self._build_array(datashape.descr, fill_value=fill_value)
        return SciDBArray(datashape, self, name)

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
        datashape = SciDBDataShape(shape, dtype, **kwargs)
        fill_value = 'random() % {0} + {1}'.format(upper - lower, lower)
        name = self._build_array(datashape.descr, fill_value=fill_value)
        return SciDBArray(datashape, self, name)

    def dot(self, A, B):
        """Compute the matrix product of A and B"""
        if (A.ndim != 2) or (B.ndim != 2):
            raise ValueError("dot requires 2-dimensional arrays")
        if A.shape[1] != B.shape[0]:
            raise ValueError("array dimensions must match for matrix product")

        name = self._next_name()
        self._execute_query('store(multiply({0},{1}),{2})'.format(A.name,
                                                                  B.name,
                                                                  name))
        schema = self._show_array(name, fmt='csv')
        datashape = SciDBDataShape.from_descr(schema)
        return SciDBArray(datashape, self, name)

    def svd(self, A, return_U=True, return_S=True, return_VT=True):
        if (A.ndim != 2):
            raise ValueError("svd requires 2-dimensional arrays")
        self._execute_query("load_library('dense_linear_algebra')")

        argdict = dict(U=return_U, S=return_S, VT=return_VT)

        # TODO: check that data type is double and chunk size is 32
        ret = []
        for arg in ['U', 'S', 'VT']:
            if argdict[arg]:
                name = self._next_name()
                self._execute_query("store(gesvd({0}, '{1}'), {2})"
                                    .format(A.name, arg, name))
                schema = self._show_array(name, fmt='csv')
                descr = SciDBDataShape.from_descr(schema)
                ret.append(SciDBArray(descr, self, name))
        return tuple(ret)

    def from_array(self, A, **kwargs):
        """Initialize a scidb array from a numpy array"""
        # TODO: make this work for other data types
        if A.dtype != 'double':
            raise NotImplementedError("from_array only implemented for double")
        dtype = 'double'
        data = A.tostring(order='C')
        filename = self._upload_bytes(A.tostring(order='C'))
        arr = self.zeros(A.shape, 'double', **kwargs)
        self._execute_query("load({0},'{1}',-1,'(double)')".format(arr.name,
                                                                   filename))
        return arr

    def toarray(self, A):
        """Convert a SciDB array to a numpy array"""
        return A.toarray()

    def from_file(self, filename, **kwargs):
        raise NotImplementedError()


class SciDBShimInterface(SciDBInterface):
    """HTTP interface to SciDB via shim [1]_

    Parameters
    ----------
    hostname : string

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
            Error = SHIM_ERROR_DICT.get(e.code, SciDBUnknownError)
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
