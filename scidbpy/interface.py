"""
Low-level interface to Scidb
"""
import urllib2


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
    """SciDBInterface Base Class"""
    pass


class SciDBShimInterface(SciDBInterface):
    """HTTP interface to SciDB via shim"""
    def __init__(self, hostname):
        self.hostname = hostname.rstrip('/')
        try:
            urllib2.urlopen(self.hostname)
        except HTTPError:
            raise ValueError("Invalid hostname: {0}".format(self.hostname))

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
