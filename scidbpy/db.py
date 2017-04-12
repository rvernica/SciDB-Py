"""
>>> connect()
DB('http://localhost:8080', None, None, None, None)

>>> print(connect())
scidb_url  = 'http://localhost:8080'
scidb_auth = None
http_auth  = None
role       = None
namespace  = None
"""

import enum
import logging
import numpy
import re
import requests
import requests.compat

import schema


class Shim(enum.Enum):
    new_session = 'new_session'
    release_session = 'release_session'
    execute_query = 'execute_query'
    read_bytes = 'read_bytes'


class DB(object):
    """SciDB Shim connection object.

    >>> DB()
    DB('http://localhost:8080', None, None, None, None)

    >>> print(DB())
    scidb_url  = 'http://localhost:8080'
    scidb_auth = None
    http_auth  = None
    role       = None
    namespace  = None
    """

    _show_query = "show('{}', 'afl')"
    _one_attr_regex = re.compile("\[\( '( [^)]+ )' \)\]\n$", re.VERBOSE)

    def __init__(
            self,
            scidb_url='http://localhost:8080',
            scidb_auth=None,
            http_auth=None,
            role=None,
            namespace=None):
        self.scidb_url = scidb_url
        self.scidb_auth = scidb_auth
        self.http_auth = http_auth
        self.role = role
        self.namespace = namespace

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__,
            self.scidb_url,
            self.scidb_auth,
            self.http_auth,
            self.role,
            self.namespace)

    def __str__(self):
        return '''\
scidb_url  = '{}'
scidb_auth = {}
http_auth  = {}
role       = {}
namespace  = {}'''.format(self.scidb_url,
                          self.scidb_auth,
                          self.http_auth,
                          self.role,
                          self.namespace)

    def _shim(self, endpoint, **kwargs):
        """Make request on Shim endpoint."""
        req = requests.get(
            requests.compat.urljoin(self.scidb_url, endpoint.value),
            params=kwargs)
        req.raise_for_status()
        return req

    def _arrays(self):
        """Download the list of SciDB arrays, i.e., 'list()'.

        >>> DB()._arrays() # doctest: +ELLIPSIS
        array([ (...
              dtype=[('name', 'O'), ('uaid', '<i8'), ('aid', '<i8'), \
('schema', 'O'), ('availability', '?'), ('temporary', '?')])
        """
        id = self._shim(Shim.new_session).text
        query = 'list()'

        # Execute 'show(...)' and Download text
        self._shim(
            Shim.execute_query,
            id=id,
            query=DB._show_query.format(query),
            save='text')
        sch_str = DB._one_attr_regex.match(
            self._shim(Shim.read_bytes, id=id, n=0).text).group(1)

        # Parse Schema
        logging.debug(sch_str)
        sch = schema.Schema.fromstring(sch_str)

        # Execute Query and Download content
        self._shim(Shim.execute_query, id=id, query=query, save=sch.atts_fmt)
        buf = self._shim(Shim.read_bytes, id=id, n=0).content

        self._shim(Shim.release_session, id=id)

        # Scan content and build (offset, size) metadata
        off = 0
        buf_meta = []
        while off < len(buf):
            meta = []
            for att in sch.atts:
                sz = att.itemsize(buf, off)
                meta.append((off, sz))
                off += sz
            buf_meta.append(meta)

        # Extract values using (offset, size) metadata
        # Create and populate  NumPy record array
        ar = numpy.empty((len(buf_meta), ), dtype=sch.atts_dtype)
        pos = 0
        for meta in buf_meta:
            ar.put((pos,),
                   tuple(att.frombytes(buf, off, sz)
                         for (att, (off, sz)) in zip(sch.atts, meta)))
            pos += 1

        return ar


connect = DB


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    import doctest
    doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)
