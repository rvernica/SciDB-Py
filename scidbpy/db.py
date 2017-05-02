"""
Connect to SciDB
----------------

Connect to SciDB using "connect()" or "DB()":

>>> db = connect()
>>> db = DB()


Display information about the "db" object:

>>> db
DB('http://localhost:8080', None, None, None, None, None)

>>> print(db)
scidb_url  = 'http://localhost:8080'
scidb_auth = None
http_auth  = None
role       = None
namespace  = None
verify     = None


Provide Shim credentials:

>>> db_ha = connect(http_auth=('foo', 'bar'))

>>> db_ha
DB('http://localhost:8080', None, ('foo', PASSWORD_PROVIDED), None, None, None)

>>> print(db_ha)
scidb_url  = 'http://localhost:8080'
scidb_auth = None
http_auth  = ('foo', PASSWORD_PROVIDED)
role       = None
namespace  = None
verify     = None

To prompt the user for the password, use:

# >>> import getpass
# >>> db_ha = connect(http_auth=('foo', getpass.getpass()))
# Password:


Use SSL:

>>> db_ssl = connect('https://localhost:8083', verify=False)

>>> print(db_ssl)
scidb_url  = 'https://localhost:8083'
scidb_auth = None
http_auth  = None
role       = None
namespace  = None
verify     = False

See Python "requests" library SSL Cert Verification section [1] for
details on the "verify" parameter.

[1] http://docs.python-requests.org/en/master/user/advanced/
    #ssl-cert-verification


Use SSL and SciDB credentials:

>>> db_sdb = connect(
...   'https://localhost:8083', scidb_auth=('foo', 'bar'), verify=False)

>>> print(db_sdb)
scidb_url  = 'https://localhost:8083'
scidb_auth = ('foo', PASSWORD_PROVIDED)
http_auth  = None
role       = None
namespace  = None
verify     = False



Access SciDB Arrays
-------------------

Access SciDB arrays using "db.arrays":

>>> iquery(db, 'create array foo<x:int64>[i=0:2]')

>>> dir(db.arrays)
... # doctest: +ELLIPSIS
[...'foo']

>>> iquery(db, 'remove(foo)')

>>> dir(db.arrays)
[]

Arrays specified explicitly are not checked:

>>> db.arrays.foo
'foo'
>>> db.arrays.bar
'bar'

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

>>> iquery(db, 'build(<x:int64 not null>[i=0:2], i)', fetch=True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0), (1, 1), (2, 2)],
      dtype=[('x', '<i8'), ('i', '<i8')])

Optionally, download only the attributes:

>>> iquery(db,
...        'build(<x:int64 not null>[i=0:2], i)',
...        fetch=True,
...        atts_only=True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0,), (1,), (2,)],
      dtype=[('x', '<i8')])


If dimension names collide with attribute names, unique dimension
names are created:

>>> iquery(db, 'apply(build(<x:int64 not null>[i=0:2], i), i, i)', fetch=True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0, 0), (1, 1, 1), (2, 2, 2)],
      dtype=[('x', '<i8'), ('i', '<i8'), ('i_1', '<i8')])


If schema is known, it can be provided to "iquery":

>>> iquery(db,
...        'build(<x:int64 not null>[i=0:2], i)',
...        fetch=True,
...        schema=Schema(None,
...                      (Attribute('x', 'int64', not_null=True),),
...                      (Dimension('i', 0, 2),)))
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0), (1, 1), (2, 2)],
      dtype=[('x', '<i8'), ('i', '<i8')])

>>> iquery(db,
...        'build(<x:int64 not null>[i=0:2], i)',
...        fetch=True,
...        atts_only=True,
...        schema=Schema.fromstring('<x:int64 not null>[i=0:2]'))
... # doctest: +NORMALIZE_WHITESPACE
array([(0,), (1,), (2,)],
      dtype=[('x', '<i8')])


Download as Pandas DataFrame:

>>> iquery(db,
...        'build(<x:int64 not null>[i=0:2], i)',
...        fetch=True,
...        as_dataframe=True)
... # doctest: +NORMALIZE_WHITESPACE
   x
i
0  0
1  1
2  2

>>> iquery(db,
...        'build(<x:int64 not null>[i=0:2], i)',
...        fetch=True,
...        atts_only=True,
...        as_dataframe=True)
... # doctest: +NORMALIZE_WHITESPACE
   x
0  0
1  1
2  2
"""

import copy
import enum
import itertools
import logging
import numpy
import pandas
import re
import requests

from .schema import Attribute, Dimension, Schema


class Shim(enum.Enum):
    new_session = 'new_session'
    release_session = 'release_session'
    execute_query = 'execute_query'
    read_bytes = 'read_bytes'


class Password_Placeholder(object):
    def __repr__(self):
        return 'PASSWORD_PROVIDED'


class DB(object):
    """SciDB Shim connection object.

    >>> DB()
    DB('http://localhost:8080', None, None, None, None, None)

    >>> print(DB())
    scidb_url  = 'http://localhost:8080'
    scidb_auth = None
    http_auth  = None
    role       = None
    namespace  = None
    verify     = None
    """

    _show_query = "show('{}', 'afl')"
    _one_attr_regex = re.compile("\[\( '( [^)]+ )' \)\]\n$", re.VERBOSE)

    def __init__(
            self,
            scidb_url='http://localhost:8080',
            scidb_auth=None,
            http_auth=None,
            role=None,
            namespace=None,
            verify=None):
        self.scidb_url = scidb_url
        self.role = role
        self.namespace = namespace
        self.verify = verify

        if http_auth:
            self._http_auth = requests.auth.HTTPDigestAuth(*http_auth)
            self.http_auth = (http_auth[0], Password_Placeholder())
        else:
            self._http_auth = self.http_auth = None

        if scidb_auth:
            if not self.scidb_url.lower().startswith('https'):
                raise Exception(
                    'SciDB credentials can only be used ' +
                    'with https connections')

            self._scidb_auth = {'user': scidb_auth[0],
                                'password': scidb_auth[1]}
            self.scidb_auth = (scidb_auth[0], Password_Placeholder())
        else:
            self._scidb_auth = self.scidb_auth = None

        self.arrays = Arrays(self)

    def __iter__(self):
        return (i for i in (
            self.scidb_url,
            self.scidb_auth,
            self.http_auth,
            self.role,
            self.namespace,
            self.verify))

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__, *self)

    def __str__(self):
        return '''\
scidb_url  = '{}'
scidb_auth = {}
http_auth  = {}
role       = {}
namespace  = {}
verify     = {}'''.format(*self)

    def iquery(self,
               query,
               fetch=False,
               atts_only=False,
               as_dataframe=False,
               schema=None):
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
                # Deep-copy schema since we might be mutating it
                sch = copy.deepcopy(schema)
            else:
                # Execute 'show(...)' and Download text
                self._shim(
                    Shim.execute_query,
                    id=id,
                    query=DB._show_query.format(query.replace("'", "\\'")),
                    save='text')
                sch_str = DB._one_attr_regex.match(
                    self._shim(Shim.read_bytes, id=id, n=0).text).group(1)

                # Parse Schema
                logging.debug(sch_str)
                sch = Schema.fromstring(sch_str)
                logging.debug(sch)

            # Unpack
            if not atts_only:
                if sch.make_dims_unique():
                    # Dimensions renamed due to collisions. Need to
                    # cast.
                    query = 'cast({}, {:h})'.format(query, sch)

                query = 'apply({}, {})'.format(
                    query,
                    ', '.join('{0}, {0}'.format(d.name) for d in sch.dims))

                sch.make_dims_attr()
                logging.debug(query)
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

            # Return NumPy array or Pandas dataframe
            if as_dataframe:
                return pandas.DataFrame.from_records(
                    ar,
                    index=[dim.name for dim in sch.dims]
                    if not atts_only else None)
            else:
                return ar

        else:                   # fetch=False
            self._shim(Shim.execute_query, id=id, query=query, release=1)

    def _shim(self, endpoint, **kwargs):
        """Make request on Shim endpoint."""
        if self._scidb_auth:
            kwargs.update(self._scidb_auth)
        req = requests.get(
            requests.compat.urljoin(self.scidb_url, endpoint.value),
            params=kwargs,
            auth=self._http_auth,
            verify=self.verify)
        req.reason = req.content
        req.raise_for_status()
        return req


class Arrays(object):
    """Access to arrays available in SciDB"""
    def __init__(self, db):
        self._db = db
        self._schema = Schema(
            atts=(Attribute('name', 'string', not_null=True),),
            dims=(Dimension('i'),))

    def __getattr__(self, name):
        return str(name)

    def __dir__(self):
        """Download the list of SciDB arrays. Use 'project(list(), name)' to
        download only names and schemas
        """
        return self._db.iquery(
            'project(list(), name)',
            fetch=True,
            atts_only=True,
            schema=self._schema)['name'].tolist()


connect = DB
iquery = DB.iquery


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    import doctest
    doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)
