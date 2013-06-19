"""
Low-level interface to Scidb
"""
import abc
import urllib2
from .scidbarray import SciDBArray, SciDBDataShape


class SciDBError(Exception): pass
class SciDBInvalidQuery(SciDBError): pass
class SciDBInvalidSession(SciDBError): pass
class SciDBEndOfFile(SciDBError): pass
class SciDBInvalidRequest(SciDBError): pass
class SciDBQueryError(SciDBError): pass
class SciDBConnectionError(SciDBError): pass
class SciDBMemoryError(SciDBError): pass
class SciDBUnknownError(SciDBError): pass
                                           

class SciDBInterface(object):
    """SciDBInterface Abstract Base Class.

    This class provides a wrapper to the low-level interface to sciDB.  The
    actual communication with the database should be implemented via the
    ``execute()`` method of subclasses.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        # Array count will facilitate the creation of unique array names
        self.array_count = 0

    def _next_name(self):
        self.array_count += 1
        return "pyarray%.4i" % self.array_count

    @abc.abstractmethod
    def _execute(self, query, response=False, max_lines=0):
        pass

    def _create_array(self, desc, name=None, fill_value=1):
        """Utility routine to create a new array

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
        """
        if name is None:
            name = self._next_name()
        self._execute("store(build({0},{1}),{2})".format(desc,
                                                         fill_value,
                                                         name))
        return name

    def _delete_array(self, name):
        """Utility routine to delete an existing array

        Parameters
        ----------
        name : string
            The name of the array to delete.  An error will be raised if
            an array with this name does not exist in the database.
        """
        self._execute("remove({0})".format(name))

    def _scan_array(self, name, max_lines=0):
        return self._execute("scan({0})".format(name), response=True,
                             max_lines=max_lines)

    def _show_array(self, name):
        return self._execute("show({0})".format(name), response=True)

    def list_arrays(self, max_lines=100):
        return self._execute("list('arrays')", response=True,
                             max_lines=max_lines)

    def ones(self, shape, dtype='double', **kwargs):
        datashape = SciDBDataShape(shape, dtype, **kwargs)
        name = self._create_array(datashape.descr, fill_value=1)
        return SciDBArray(datashape, self, name)

    def zeros(self, shape, dtype='double', **kwargs):
        datashape = SciDBDataShape(shape, dtype, **kwargs)
        name = self._create_array(datashape.descr, fill_value=0)
        return SciDBArray(datashape, self, name)

    def random(self, shape, dtype='double', **kwargs):
        datashape = SciDBDataShape(shape, dtype, **kwargs)
        name = self._create_array(datashape.descr,
                                  fill_value='random() / 2147483647.0')
        return SciDBArray(datashape, self, name)

    def randint(self, upper, shape, dtype='uint32', **kwargs):            
        datashape = SciDBDataShape(shape, dtype, **kwargs)
        name = self._create_array(datashape.descr,
                                  fill_value='random() % {0}'.format(upper))
        return SciDBArray(datashape, self, name)

    def dot(self, A, B):
        if (A.ndim != 2) or (B.ndim != 2):
            raise ValueError("dot requires 2-dimensional arrays")
        if A.shape[1] != B.shape[0]:
            raise ValueError("array dimensions must match for matrix product")
        datashape = SciDBDataShape((A.shape[0], B.shape[1]), A.dtype)
        name = self._next_name()

        # TODO: make sure datashape matches that of the new array.
        #       How do we do this?
        self._execute('store(multiply({0},{1}),{2})'.format(A.name, B.name,
                                                            name))
        return SciDBArray(datashape, self, name)


class SciDBShimInterface(SciDBInterface):
    """HTTP interface to SciDB via shim
    
    Parameters
    ----------
    hostname : string
    session_id : integer
    """
    def __init__(self, hostname, session_id=None):
        self.hostname = hostname.rstrip('/')
        self.session_id = session_id
        try:
            urllib2.urlopen(self.hostname)
        except HTTPError:
            raise ValueError("Invalid hostname: {0}".format(self.hostname))
        SciDBInterface.__init__(self)

    def _execute(self, query, response=False, max_lines=0):
        session_id = self._new_session()
        if response:
            self._execute_query(session_id, query, save='csv', release=False)
            result = self._read_lines(session_id, max_lines)
            self._release_session(session_id)
        else:
            self._execute_query(session_id, query, release=True)
            result = None
        return result

    def _url(self, keyword, **kwargs):
        url = self.hostname + '/' + keyword
        if kwargs:
            url += '?' + '&'.join(['{0}={1}'.format(key, val)
                                   for key, val in kwargs.iteritems()])
        return url

    def _urlopen(self, url):
        try:
            return urllib2.urlopen(url)
        except urllib2.HTTPError as e:
            self._handle_error(e.code, e.read())

    def _handle_error(self, code, message=''):
        # Any error kills the session
        self.session_id = None

        if code == 400:
            raise SciDBInvalidQuery(message)
        elif code == 404:
            raise SciDBInvalidSession(message)
        elif code == 410:
            raise SciDBEndOfFile(message)
        elif code == 414:
            raise SciDBInvalidRequest(message)
        elif code == 500:
            raise SciDBQueryError(message)
        elif code == 503:
            raise SciDBConnectionError(message)
        elif code == 507:
            raise SciDBMemoryError(message)
        else:
            raise SciDBUnknownError("HTTP {0}: {1}".format(code, message))

    def _new_session(self):
        """Request a new HTTP session from the service"""
        url = self._url('new_session')
        result = self._urlopen(url)
        session_id = int(result.read())
        return session_id

    def _release_session(self, session_id):
        url = self._url('release_session', id=session_id)
        result = self._urlopen(url)

    def _execute_query(self, session_id, query, save=None, release=False):
        url = self._url('execute_query',
                        id=session_id,
                        query=urllib2.quote(query),
                        release=int(bool(release)))
        if save is not None:
            url += "&save={0}".format(save)

        result = self._urlopen(url)
        query_id = result.read()
        return query_id

    def _cancel(self, session_id):
        url = self._url('cancel', id=session_id)
        result = self._urlopen(url)
        
    def _read_lines(self, session_id, n):
        url = self._url('read_lines', id=session_id, n=n)
        result = self._urlopen(url)
        text_result = result.read()
        return text_result

    def _read_bytes(self, session_id, n):
        url = self._url('read_lines', id=session_id, n=n)
        result = self._urlopen(url)
        bytes_result = result.read()
        return bytes_result

    def _upload_file(self, session_id, filename, data):
        import requests
        url = self._url('upload_file', id=session_id)
        result = requests.post(url, files={filename:data})
        scidb_filename = result.text
        return scidb_filename
