"""
>>> db = connect()

>>> db
DB('http://localhost:8080', None, None, None, None)

>>> print(db)
scidb_url  = 'http://localhost:8080'
scidb_auth = None
http_auth  = None
role       = None
namespace  = None


>>> iquery(db, 'store(build(<x:int64>[i=0:2], i), foo)')
>>> iquery(db, 'scan(foo)')

>>> iquery(db, 'scan(foo)', fetch=True) # doctest: +NORMALIZE_WHITESPACE
array([((255, 0), 0), ((255, 1), 1), ((255, 2), 2)],
      dtype=[('x', [('null', 'u1'), ('val', '<i8')]), ('i', '<i8')])

>>> iquery(db, 'scan(foo)', fetch=True, atts_only=True)
... # doctest: +NORMALIZE_WHITESPACE
array([((255, 0),), ((255, 1),), ((255, 2),)],
      dtype=[('x', [('null', 'u1'), ('val', '<i8')])])

>>> iquery(db, 'remove(foo)')


>>> iquery(db, 'build(<x:int64 not null>[i=0:2], i)', True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0), (1, 1), (2, 2)],
      dtype=[('x', '<i8'), ('i', '<i8')])

>>> iquery(db, 'build(<x:int64 not null>[i=0:2], i)', True, True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0,), (1,), (2,)],
      dtype=[('x', '<i8')])

>>> iquery(db,                                                                \
           'build(<x:int64 not null>[i=0:2], i)',                             \
           fetch=True,                                                        \
           schema=Schema('build',                                             \
                         (Attribute('x', 'int64', not_null=True),),           \
                         (Dimension('i', 0, 2),)))
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0), (1, 1), (2, 2)],
      dtype=[('x', '<i8'), ('i', '<i8')])

>>> iquery(db,                                                                \
           'build(<x:int64 not null>[i=0:2], i)',                             \
           fetch=True,                                                        \
           atts_only=True,                                                    \
           schema=Schema.fromstring('build<x:int64 not null>[i=0:2]'))
... # doctest: +NORMALIZE_WHITESPACE
array([(0,), (1,), (2,)],
      dtype=[('x', '<i8')])
"""

import enum
import itertools
import logging
import numpy
import re
import requests
import requests.compat

from schema import Attribute, Dimension, Schema


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

    def iquery(self, query, fetch=False, atts_only=False, schema=None):
        """Execute query in SciDB"""
        id = self._shim(Shim.new_session).text

        if fetch:
            # Use provided schema or get schema from SciDB
            if schema:
                sch = schema
            else:
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
                sch = Schema.fromstring(sch_str)
                logging.debug(sch)

            # Unpack
            if not atts_only:
                query = 'apply({}, {})'.format(
                    query,
                    ', '.join('{0}, {0}'.format(d.name) for d in sch.dims))

                sch = Schema(
                    sch.name,
                    itertools.chain(
                        sch.atts,
                        (Attribute(d.name, 'int64', True)
                         for d in sch.dims)),
                    sch.dims)
                logging.debug(sch)

            # Execute Query and Download content
            self._shim(
                Shim.execute_query, id=id, query=query, save=sch.atts_fmt)
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
            # Create and populate NumPy record array
            ar = numpy.empty((len(buf_meta),), dtype=sch.atts_dtype)
            pos = 0
            for meta in buf_meta:
                ar.put((pos,),
                       tuple(att.frombytes(buf, off, sz)
                             for (att, (off, sz)) in zip(sch.atts, meta)))
                pos += 1

            return ar

        else:                   # fetch=False
            self._shim(Shim.execute_query, id=id, query=query)
            self._shim(Shim.release_session, id=id)

    def _shim(self, endpoint, **kwargs):
        """Make request on Shim endpoint."""
        req = requests.get(
            requests.compat.urljoin(self.scidb_url, endpoint.value),
            params=kwargs)
        req.raise_for_status()
        return req

    def _arrays(self):
        """Download the list of SciDB arrays, i.e., 'list()'.

        >>> DB()._arrays() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        array([...],
              dtype=[('name', 'O'), ('uaid', '<i8'), ('aid', '<i8'), \
                     ('schema', 'O'), ('availability', '?'), \
                     ('temporary', '?')])
        """
        return self.iquery('list()', True, True)


connect = DB
iquery = DB.iquery


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    import doctest
    doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)
