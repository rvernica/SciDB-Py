"""
Quick Start
===========

>>> from scidbpy import connect, iquery

>>> db = connect()
>>> iquery(db, 'store(build(<x:int64>[i=0:2], i), foo)')

>>> db.arrays.foo[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

>>> import numpy
>>> ar = db.input(upload_data=numpy.arange(3)).store()
>>> del ar

>>> db.remove(db.arrays.foo)


Connect to SciDB
----------------

Connect to SciDB using "connect()" or "DB()":


>>> db = connect()


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

>>> iquery(db, 'store(build(<x:int64>[i=0:2], i), foo)')

>>> dir(db.arrays)
... # doctest: +ELLIPSIS
[...'foo']

>>> dir(db.arrays.foo)
... # doctest: +ELLIPSIS
[...'i', ...'x']

>>> db.arrays.foo[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

>>> iquery(db, 'remove(foo)')

>>> dir(db.arrays)
[]

Arrays specified explicitly are not checked:

>>> print(db.arrays.foo)
foo
>>> print(db.arrays.bar)
bar

In IPython, you can use <TAB> for auto-completion of array names,
array dimensions, and array attributes:

# In []: db.arrays.<TAB>
# In []: db.arrays.foo
# In []: db.arrays.foo.<TAB>
# In []: db.arrays.foo.x


Query SciDB
-----------

Use "iquery" to execute queries against SciDB:

>>> iquery(db, 'store(build(<x:int64>[i=0:2], i), foo)')


Dwonload Data from SciDB
------------------------

Use "iquery" to download array data:

>>> iquery(db, 'scan(foo)', fetch=True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

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
      dtype=[('i', '<i8'), ('x', '<i8')])

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
      dtype=[('i_1', '<i8'), ('x', '<i8'), ('i', '<i8')])


If schema is known, it can be provided to "iquery":

>>> from scidbpy import Schema
>>> iquery(db,
...        'build(<x:int64 not null>[i=0:2], i)',
...        fetch=True,
...        schema=Schema(None,
...                      (Attribute('x', 'int64', not_null=True),),
...                      (Dimension('i', 0, 2),)))
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0), (1, 1), (2, 2)],
      dtype=[('i', '<i8'), ('x', '<i8')])

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
...        'build(<x:int64>[i=0:2], i)',
...        fetch=True,
...        as_dataframe=True)
... # doctest: +NORMALIZE_WHITESPACE
   i x
0  0 0.0
1  1 1.0
2  2 2.0

>>> iquery(db,
...        'build(<x:int64>[i=0:2], i)',
...        fetch=True,
...        atts_only=True,
...        as_dataframe=True)
... # doctest: +NORMALIZE_WHITESPACE
   x
0  0.0
1  1.0
2  2.0

>>> iquery(db,
...        'build(<x:int64>[i=0:2], i)',
...        fetch=True,
...        atts_only=True,
...        as_dataframe=True,
...        dataframe_promo=False)
... # doctest: +NORMALIZE_WHITESPACE
   x
0  (255, 0)
1  (255, 1)
2  (255, 2)


Upload Data to SciDB
--------------------

Data can be uploaded using the "iquery" function, by providing an
"upload_data" argument.

Provide SciDB input/store/insert/load query, NumPy array, and,
optional, schema. If the schema is missing, it is inferred from the
array dtype. If the format is missing from the query string, it is
inferred from schema:

>>> db.iquery("store(input(<x:int64>[i], '{fn}', 0, '{fmt}'), foo)",
...           upload_data=numpy.arange(3))

>>> db.arrays.foo[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

>>> db.iquery("insert(input({sch}, '{fn}', 0, '(int64)'), foo)",
...           upload_data=numpy.arange(3))

>>> db.iquery("load(foo, '{fn}', 0, '{fmt}')",
...           upload_data=numpy.arange(3),
...           upload_schema=Schema.fromstring('<x:int64 not null>[i]'))

Provide SciDB input/store/insert/load query and binary data. The query
string needs to contain the format of the binary data:

>>> db.iquery("store(input(<x:int64>[i], '{fn}', 0, '(int64)'), foo)",
...           upload_data=numpy.arange(3).tobytes())

>>> db.iquery("insert(input(foo, '{fn}', 0, '(int64)'), foo)",
...           upload_data=numpy.arange(3).tobytes())

>>> db.iquery("load(foo, '{fn}', 0, '(int64)')",
...           upload_data=numpy.arange(3).tobytes())

A binary or text file-like object can be used to specify the upload
data. The content of the file has to be in one of the supported SciDB
formats. A matching format specification has to be provided as well:

>>> with open('array.bin', 'wb') as file:
...     n = file.write(numpy.arange(3).tobytes())

>>> db.iquery("load(foo, '{fn}', 0, '(int64)')",
...           upload_data=open('array.bin', 'rb'))

>>> with open('array.csv', 'w') as file:
...     n = file.write('1\\n2\\n3\\n')

>>> db.iquery("load(foo, '{fn}', 0, 'CSV')",
...           upload_data=open('array.csv', 'r'))

>>> import os
>>> os.remove('array.bin')
>>> os.remove('array.csv')

>>> db.remove(db.arrays.foo)


Use SciDB Operators
-------------------

>>> dir(db)
... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
[...'aggregate',
 ...'apply',
 ...
 ...'xgrid']

>>> db.apply
... # doctest: +NORMALIZE_WHITESPACE
Operator(db=DB('http://localhost:8080', None, None, None, None, None),
         name='apply',
         args=[])

>>> print(db.apply)
apply()

>>> db.missing
Traceback (most recent call last):
    ...
AttributeError: 'DB' object has no attribute 'missing'

In IPython, you can use <TAB> for auto-completion of operator names:

# In []: db.<TAB>
# In []: db.apply

>>> db.create_array('foo', '<x:int64>[i]')
>>> dir(db.arrays)
... # doctest: +ELLIPSIS
[...'foo']

>>> db.remove(db.arrays.foo)
>>> dir(db.arrays)
[]

>>> db.build('<x:int8 not null>[i=0:2]', 'i + 10')[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 10), (1, 11), (2, 12)],
      dtype=[('i', '<i8'), ('x', 'i1')])

>>> db.build('<x:int8 not null>[i=0:2]', 'i + 10').fetch(as_dataframe=True)
   i   x
0  0  10
1  1  11
2  2  12

>>> db.build('<x:int8 not null>[i=0:2]', 'i + 10').apply('y', 'x - 5')[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 10, 5), (1, 11, 6), (2, 12, 7)],
      dtype=[('i', '<i8'), ('x', 'i1'), ('y', '<i8')])

>>> db.build('<x:int8 not null>[i=0:2]', 'i + 10').store('foo')
Array(DB('http://localhost:8080', None, None, None, None, None), 'foo')

>>> db.scan(db.arrays.foo)[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 10), (1, 11), (2, 12)],
      dtype=[('i', '<i8'), ('x', 'i1')])

>>> db.apply(db.arrays.foo, 'y', db.arrays.foo.x + 1)[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 10, 11), (1, 11, 12), (2, 12, 13)],
      dtype=[('i', '<i8'), ('x', 'i1'), ('y', '<i8')])

>>> db.remove(db.arrays.foo)

Input and load operators can be used to upload data. If the schema and
the format are not specified, they are inferred from the uploaded
data. If an array name is not specified for the store operator, an
array name is generated. Arrays with generated names are removed when
the returned Array object is garbage collected. This behavior can be
changed by specifying "gc=False" to the store operator:

>>> ar = db.input(upload_data=numpy.arange(3)).store()
>>> ar
... # doctest: +ELLIPSIS
Array(DB('http://localhost:8080', None, None, None, None, None), 'py_...')
>>> del ar

>>> db.input('<x:int64>[i]', upload_data=numpy.arange(3))[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

>>> db.input('<x:int64>[i]', upload_data=numpy.arange(3)).store(db.arrays.foo)
... # doctest: +NORMALIZE_WHITESPACE
Array(DB('http://localhost:8080', None, None, None, None, None), 'foo')

>>> db.load(db.arrays.foo, upload_data=numpy.arange(3))

>>> db.input('<x:int64>[j]', upload_data=numpy.arange(3, 6)
...  ).apply('i', 'j + 3'
...  ).redimension(db.arrays.foo
...  ).insert(db.arrays.foo)

>>> db.arrays.foo[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2)), (3, (255, 3)),
       (4, (255, 4)), (5, (255, 5))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

>>> db.input('<i:int64 not null, x:int64>[j]', upload_data=db.arrays.foo[:]
...  ).redimension(db.arrays.foo
...  ).store('bar')
Array(DB('http://localhost:8080', None, None, None, None, None), 'bar')

>>> numpy.all(db.arrays.bar[:] == db.arrays.foo[:])
True

>>> db.remove(db.arrays.foo)
>>> db.remove(db.arrays.bar)

For files already available on the server the "input" or "load"
operators can be invoked with the full set of parameters supported by
SciDB. Parameters that need to be quoted in SciDB need to be
double-quoted in SciDB-Py. For example:

# >>> db.load('foo', "'/data.csv'", 0, "'CSV'")
"""

from .db import connect, iquery, Array
from .schema import Attribute, Dimension, Schema

__version__ = '16.9'
