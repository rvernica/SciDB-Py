"""
Connect to SciDB
----------------

Connect to SciDB using "connect()" or "DB()":

>>> db = connect()
>>> db = DB()


Display information about the "db" object:

>>> db
DB('http://localhost:8080', None, None, None, None)

>>> print(db)
scidb_url  = 'http://localhost:8080'
scidb_auth = None
http_auth  = None
role       = None
namespace  = None


Access SciDB Arrays
-------------------

Access SciDB arrays using "db.arrays":

>>> iquery(db, 'create array foo<x:int64>[i=0:2]')
>>> db.arrays.foo
... # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
Schema(...'foo',
       (Attribute(...'x', ...'int64', False, None, None),),
       (Dimension(...'i', 0, 2, 0, ...'*'),))

>>> print(db.arrays.foo)
foo<x:int64> [i=0:2:0:*]

>>> iquery(db, 'remove(foo)')
>>> db.arrays.foo
Traceback (most recent call last):
  ...
KeyError: 'foo'

In IPython, you can use <TAB> for auto-completion of array names:

# In []: db.arrays.<TAB>
# In []: db.arrays.foo


Use "iquery" function
---------------------

Use "iquery" to execute queries:

>>> iquery(db, 'store(build(<x:int64>[i=0:2], i), foo)')


Use "iquery" to download array data:

>>> iquery(db, 'scan(foo)', fetch=True)
... # doctest: +NORMALIZE_WHITESPACE
array([((255, 0), 0), ((255, 1), 1), ((255, 2), 2)],
      dtype=[('x', [('null', 'u1'), ('val', '<i8')]), ('i', '<i8')])

Optionally, download only the attributes:

>>> iquery(db, 'scan(foo)', fetch=True, atts_only=True)
... # doctest: +NORMALIZE_WHITESPACE
array([((255, 0),), ((255, 1),), ((255, 2),)],
      dtype=[('x', [('null', 'u1'), ('val', '<i8')])])

>>> iquery(db, 'remove(foo)')


Download operator output directly:

>>> iquery(db, 'build(<x:int64 not null>[i=0:2], i)', True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0), (1, 1), (2, 2)],
      dtype=[('x', '<i8'), ('i', '<i8')])

Optionally, download only the attributes:

>>> iquery(db, 'build(<x:int64 not null>[i=0:2], i)', True, True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0,), (1,), (2,)],
      dtype=[('x', '<i8')])


If schema is known, it can be provided to "iquery":

>>> iquery(db,
...        'build(<x:int64 not null>[i=0:2], i)',
...        fetch=True,
...        schema=Schema('build',
...                      (Attribute('x', 'int64', not_null=True),),
...                      (Dimension('i', 0, 2),)))
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0), (1, 1), (2, 2)],
      dtype=[('x', '<i8'), ('i', '<i8')])

>>> iquery(db,
...        'build(<x:int64 not null>[i=0:2], i)',
...        fetch=True,
...        atts_only=True,
...        schema=Schema.fromstring('build<x:int64 not null>[i=0:2]'))
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

        self._update_arrays()

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

    def iquery(self,
               query,
               fetch=False,
               atts_only=False,
               schema=None,
               update=True):
        """Execute query in SciDB

        >>> DB().iquery('build(<x:int64>[i=0:1; j=0:1], i + j)', fetch=True)
        ... # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        array([((255, 0), 0, 0),
               ((255, 1), 0, 1),
               ((255, 1), 1, 0),
               ((255, 2), 1, 1)],
              dtype=[('x', [('null', 'u1'), ('val', '<i8')]),
                     ('i', '<i8'), ('j', '<i8')])
        """
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
                        (Attribute(d.name, 'int64', not_null=True)
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

            ret = ar

        else:                   # fetch=False
            self._shim(Shim.execute_query, id=id, query=query, release=1)
            ret = None

        if update:
            self._update_arrays()
        return ret

    def _shim(self, endpoint, **kwargs):
        """Make request on Shim endpoint."""
        req = requests.get(
            requests.compat.urljoin(self.scidb_url, endpoint.value),
            params=kwargs)
        req.raise_for_status()
        return req

    def _update_arrays(self):
        """Download the list of SciDB arrays. Use 'project(list(), name,
        schema)' to download only names and schemas

        """
        ar = self.iquery(
            'project(list(), name, schema)',
            fetch=True,
            atts_only=True,
            schema=Schema(
                'list',
                (Attribute('name', 'string', not_null=True),
                 Attribute('schema', 'string', not_null=True)),
                (Dimension('i'),)),
            update=False)
        self.arrays = Arrays(ar)


class Arrays(object):
    """Access to arrays available in SciDB"""
    def __init__(self, arrays):
        self.array_map = dict(
            ((n, Schema.fromstring(s))
             for (n, s) in zip(arrays['name'], arrays['schema'])))

    def __getattr__(self, name):
        return self.array_map[name]

    def __dir__(self):
        return self.array_map.keys()


connect = DB
iquery = DB.iquery


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    import doctest
    doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)
