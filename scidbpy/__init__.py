"""Quick Start
===========

Install the package with::

  $ pip install scidb-py

Connect to SciDB and run a query:

>>> from scidbpy import connect
>>> db = connect('http://localhost:8080')
>>> db.iquery('store(build(<x:int64>[i=0:2], i), foo)')

Download data from SciDB:

>>> db.arrays.foo[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

Upload data to SciDB and create an array:

>>> import numpy
>>> ar = db.input(upload_data=numpy.arange(3)).store()
>>> print(ar)
... # doctest: +ELLIPSIS
py_..._1

Run a query with chained operators and download the resulting array:

>>> db.join(ar, 'foo').apply('j', ar.i + 1)[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0, (255, 0), 1), (1, 1, (255, 1), 2), (2, 2, (255, 2), 3)],
      dtype=[('i', '<i8'),
             ('x', '<i8'),
             ('x_1', [('null', 'u1'), ('val', '<i8')]),
             ('j', '<i8')])

Cleanup:

>>> db.remove(db.arrays.foo)
>>> del ar


Requirements
============

SciDB ``16.9`` with Shim

Python ``2.7.x``, ``3.4.x``, ``3.5.x``, ``3.6.x`` or newer.

Required Python packages::

  backports.weakref
  enum34
  numpy
  pandas
  requests
  six


Installation
============

Install latest release::

  $ pip install scidb-py

Install development version from GitHub::

  $ pip install git+http://github.com/paradigm4/scidb-py.git


Connect to SciDB
================

Connect to SciDB using :func:`connect()<scidbpy.db.connect>`:

>>> from scidbpy import connect
>>> db = connect()
>>> db = connect('http://localhost:8080')

``connect()`` is aliased to the constructor of the
:class:`DB<scidbpy.db.DB>` class. See :meth:`DB()<scidbpy.db.DB>` for
the complete set of arguments that can be provided to
``connect``. ``http://localhost:8080`` is the default connection URL
is none is provided.

Display information about the ``db`` object:

>>> db
DB('http://localhost:8080', None, None, None, None)

>>> print(db)
scidb_url  = http://localhost:8080
scidb_auth = None
http_auth  = None
namespace  = None
verify     = None

Advanced Connection
-------------------

Provide `Shim <https://github.com/Paradigm4/shim>`_ credentials:

>>> db = connect(http_auth=('foo', 'bar'))

>>> db
DB('http://localhost:8080', None, ('foo', PASSWORD_PROVIDED), None, None)

>>> print(db)
scidb_url  = http://localhost:8080
scidb_auth = None
http_auth  = ('foo', PASSWORD_PROVIDED)
namespace  = None
verify     = None

To prompt the user for the password, use:

>>> import getpass
>>> db = connect(http_auth=('foo', getpass.getpass()))
... # doctest: +SKIP
Password:


Use SSL:

>>> db_ssl = connect('https://localhost:8083', verify=False)

>>> print(db_ssl)
scidb_url  = https://localhost:8083
scidb_auth = None
http_auth  = None
namespace  = None
verify     = False

See Python `requests <http://docs.python-requests.org/en/master/>`_
library `SSL Cert Verification
<http://docs.python-requests.org/en/master/user/advanced/
#ssl-cert-verification>`_ section for details on the ``verify``
argument. ``verify=False`` disables SSL certificate
verification. Warnings about the unverified HTTPS requests are
displayed. The warnings can be disables as well by either setting
``PYTHONWARNINGS="ignore:Unverified HTTPS request"`` in the
environment before starting Python or by doing:

>>> import requests
>>> requests.packages.urllib3.disable_warnings(
...   requests.packages.urllib3.exceptions.InsecureRequestWarning)


Use SSL and SciDB credentials:

>>> db_ssl = connect(
...   'https://localhost:8083', scidb_auth=('foo', 'bar'), verify=False)

>>> print(db_ssl)
scidb_url  = https://localhost:8083
scidb_auth = ('foo', PASSWORD_PROVIDED)
http_auth  = None
namespace  = None
verify     = False


By default, the ``connect`` function queries SciDB for the list of
available operators. This list is used for easy access to the SciDB
operators, see the *SciDB Operators* section below. As a consequence
the connection to SciDB is verified and a small delay might
occur. This behavior can be disabled using the ``no_ops=True``
parameter. Accessing the SciDB operators as described in *SciDB
Operators* will not be possible until the ``load_ops()`` function is
called on the ``DB`` instance:

>>> db_no_ops = connect(no_ops=True)
>>> db_no_ops.scan
Traceback (most recent call last):
    ...
AttributeError: 'DB' object has no attribute 'scan'

No query has been issued to SciDB yet.

>>> db_no_ops.load_ops()
>>> db_no_ops.scan
... # doctest: +NORMALIZE_WHITESPACE
Operator(db=DB('http://localhost:8080', None, None, None, None),
         name='scan',
         args=[])


SciDB Arrays
============

SciDB arrays can be accessed using ``DB.arrays``:

>>> db = connect()
>>> db.iquery('store(build(<x:int64>[i=0:2], i), foo)')

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

>>> db.iquery('remove(foo)')

>>> dir(db.arrays)
[]

Arrays specified explicitly are not checked:

>>> print(db.arrays.foo)
foo
>>> print(db.arrays.bar)
bar

In IPython, you can use <TAB> for auto-completion of array names,
array dimensions, and array attributes::

    In [1]: db.arrays.<TAB>
    In [1]: db.arrays.foo
    In [2]: db.arrays.foo.<TAB>
    In [2]: db.arrays.foo.x


SciDB Operators
===============

At connection time, the library downloads the list of available SciDB
operators and macros and makes them available through the ``DB`` class
instance:

>>> dir(db)
... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
[...'aggregate',
 ...'apply',
 ...
 ...'xgrid']

>>> db.apply
... # doctest: +NORMALIZE_WHITESPACE
Operator(db=DB('http://localhost:8080', None, None, None, None),
         name='apply',
         args=[])

>>> print(db.apply)
apply()

>>> db.missing
Traceback (most recent call last):
    ...
AttributeError: 'DB' object has no attribute 'missing'

In IPython, you can use <TAB> for auto-completion of operator names::

    In [1]: db.<TAB>
    In [1]: db.apply

The operators can be execute immediately or can be lazy and executed
at a later time. Operators that return arrays are lazy operators (e.g.,
``apply``, ``scan``, etc.). Operator which do not return arrays
execute immediately (e.g., ``create_array``, ``remove``, etc.).

>>> db.create_array('foo', '<x:int64>[i]')
>>> dir(db.arrays)
... # doctest: +ELLIPSIS
[...'foo']

>>> db.remove(db.arrays.foo)
>>> dir(db.arrays)
[]

The list of available operators is re-loaded automatically when
a ``load_library`` query is issued:

>>> db.load_library('limit')
... # doctest: +SKIP
>>> 'limit' in dir(db)
... # doctest: +SKIP
True

A similar functionality is **not** implemented for ``unload_library``
operator. The ``unload_library`` operator requires a SciDB restart,
which makes re-loading the list of operators not
practical. Nevertheless, one can trigger the re-loading manually after
SciDB restart without creating a new ``DB`` instance:

>>> db.iquery("unload_library('limit')")
... # doctest: +SKIP

After SciDB restart:

>>> db.load_ops()
>>> 'limit' in dir(db)
... # doctest: +SKIP
False


Download Data from SciDB
------------------------

>>> db.build('<x:int8 not null>[i=0:2]', 'i + 10')[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 10), (1, 11), (2, 12)],
      dtype=[('i', '<i8'), ('x', 'i1')])

>>> db.build('<x:int8 not null>[i=0:2]', 'i + 10').fetch(as_dataframe=True)
   i   x
0  0  10
1  1  11
2  2  12

>>> db.build('<x:int8 not null>[i=0:2]', 'i + 10').fetch(
...     atts_only=True,
...     as_dataframe=True)
    x
0  10
1  11
2  12

>>> db.build('<x:int8 not null>[i=0:2]', 'i + 10').apply('y', 'x - 5')[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 10, 5), (1, 11, 6), (2, 12, 7)],
      dtype=[('i', '<i8'), ('x', 'i1'), ('y', '<i8')])

>>> db.build('<x:int8 not null>[i=0:2]', 'i + 10').store('foo')
Array(DB('http://localhost:8080', None, None, None, None), 'foo')

>>> db.scan(db.arrays.foo)[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 10), (1, 11), (2, 12)],
      dtype=[('i', '<i8'), ('x', 'i1')])

>>> db.apply(db.arrays.foo, 'y', db.arrays.foo.x + 1)[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 10, 11), (1, 11, 12), (2, 12, 13)],
      dtype=[('i', '<i8'), ('x', 'i1'), ('y', '<i8')])

>>> db.remove(db.arrays.foo)


Upload Data to SciDB
--------------------

``input`` and ``load`` operators can be used to upload data. An upload
schema can also be provided. If the resulting array or schema is not
provided, it can be generated from the upload data or upload
schema. If the upload format is not provided, it can be constructed
from the upload schema, upload data, or resulting array schema.

If an array name is not specified for the ``store`` operator, an array
name is generated. Arrays with generated names are removed when the
returned Array object is garbage collected. This behavior can be
changed by specifying the ``gc=False`` argument to the store operator.

>>> ar = db.input(upload_data=numpy.arange(3)).store()
>>> ar
... # doctest: +ELLIPSIS
Array(DB('http://localhost:8080', None, None, None, None), 'py_...')
>>> del ar

>>> db.input('<x:int64>[i]', upload_data=numpy.arange(3))[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

>>> db.input('<x:int64>[i]', upload_data=numpy.arange(3)).store(db.arrays.foo)
Array(DB('http://localhost:8080', None, None, None, None), 'foo')

>>> db.load(db.arrays.foo, upload_data=numpy.arange(3))
Array(DB('http://localhost:8080', None, None, None, None), 'foo')

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
Array(DB('http://localhost:8080', None, None, None, None), 'bar')

>>> numpy.all(db.arrays.bar[:] == db.arrays.foo[:])
True

>>> buf = numpy.array([bytes([10, 20, 30])], dtype='object')

>>> db.input('<b:binary not null>[i]', upload_data=buf).store('taz')
Array(DB('http://localhost:8080', None, None, None, None), 'taz')

>>> db.load('taz',
...         upload_data=buf,
...         upload_schema=Schema.fromstring('<b:binary not null>[i]'))
Array(DB('http://localhost:8080', None, None, None, None), 'taz')

For files already available on the server the ``input`` or ``load``
operators can be invoked with the full set of arguments supported by
SciDB. Arguments that need to be quoted in SciDB need to be
double-quoted in SciDB-Py. For example:

>>> db.load('foo', "'/data.csv'", 0, "'CSV'")
... # doctest: +SKIP

>>> for ar in ['foo', 'bar', 'taz']: db.remove(ar)

The iquery Function
===================

Use the :meth:`DB.iquery()<scidbpy.db.DB.iquery>` function to execute
literal queries against SciDB:

>>> db.iquery('store(build(<x:int64>[i=0:2], i), foo)')


Download Data from SciDB
------------------------

The ``iquery`` function can be used to download data from SciDB by
specifying the ``fetch=True`` argument:

>>> db.iquery('scan(foo)', fetch=True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

To avoid downloading the dimension information and only download the
attributes, use the ``atts_only=True`` argument:

>>> db.iquery('scan(foo)', fetch=True, atts_only=True)
... # doctest: +NORMALIZE_WHITESPACE
array([((255, 0),), ((255, 1),), ((255, 2),)],
      dtype=[('x', [('null', 'u1'), ('val', '<i8')])])

>>> db.iquery('remove(foo)')


Download operator output directly:

>>> db.iquery('build(<x:int64 not null>[i=0:2], i)',
...           fetch=True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0), (1, 1), (2, 2)],
      dtype=[('i', '<i8'), ('x', '<i8')])

>>> db.iquery('build(<x:int64 not null>[i=0:2], i)',
...           fetch=True,
...           atts_only=True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0,), (1,), (2,)],
      dtype=[('x', '<i8')])


If dimension names collide with attribute names, unique dimension
names are created:

>>> db.iquery('apply(build(<x:int64 not null>[i=0:2], i), i, i)', fetch=True)
... # doctest: +NORMALIZE_WHITESPACE
array([(0, 0, 0), (1, 1, 1), (2, 2, 2)],
      dtype=[('i_1', '<i8'), ('x', '<i8'), ('i', '<i8')])


If schema is known, it can be provided to ``iquery`` using the
``schema`` argument. This speeds up the execution as ``iquery`` does
not need to issue a ``show()`` query first in order to determine the
schema:

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

Data can be uploaded using the ``iquery`` function by providing an
``upload_data`` argument. A file name placeholder needs to be provided
as part of the SciDB query string. The upload array schema and data
format can be provided explicitly or as placeholders in the query
string. The placeholders are replaced with the explicit values by the
``iquery`` function, before the query is sent to SciDB.

The SciDB query placeholders are:

* ``'{fn}'``: **mandatory** placeholder which is replaced with the
  file name of the server file where the uploaded data is stored. It
  has to be *quoted* with single quotes in the query string.

* ``{sch}``: *optional* placeholder which is replaced with the upload
  array schema. It does *not* need to be quoted.

* ``'{fmt}'``: *optional* placeholder which is replaced with the
  upload array format.  It has to be *quoted* with single quotes in
  the query string.

See examples in the following subsections.

Upload NumPy Arrays
^^^^^^^^^^^^^^^^^^^

Provide a SciDB ``input``, ``store``, ``insert``, or ``load`` query
and a NumPy array. If the schema or format are provided as
placeholders, the upload data *dtype* or upload schema is used to
populate these placeholders.

>>> db.iquery("store(input(<x:int64>[i], '{fn}', 0, '{fmt}'), foo)",
...           upload_data=numpy.arange(3))

>>> db.arrays.foo[:]
... # doctest: +NORMALIZE_WHITESPACE
array([(0, (255, 0)), (1, (255, 1)), (2, (255, 2))],
      dtype=[('i', '<i8'), ('x', [('null', 'u1'), ('val', '<i8')])])

>>> db.iquery("insert(input({sch}, '{fn}', 0, '(int64)'), foo)",
...           upload_data=numpy.arange(3))

Optionally, a ``Schema`` object can be used to specify the upload
schema using the ``upload_schema`` argument:

>>> db.iquery("load(foo, '{fn}', 0, '{fmt}')",
...           upload_data=numpy.arange(3),
...           upload_schema=Schema.fromstring('<x:int64 not null>[i]'))


Upload Binary Data
^^^^^^^^^^^^^^^^^^

Provide a SciDB ``input``, ``store``, ``insert``, or ``load`` query
and binary data. The schema of the upload data needs to be provided
either explicitly in the query string or using the ``upload_schema``
argument. If the schema is not provides using the ``upload_schema``
argument, the format needs to be provided explicitly in the query
string:

>>> db.iquery("store(input({sch}, '{fn}', 0, '{fmt}'), foo)",
...           upload_data=numpy.arange(3).tobytes(),
...           upload_schema=Schema.fromstring('<x:int64 not null>[i]'))

>>> db.iquery("insert(input(foo, '{fn}', 0, '(int64)'), foo)",
...           upload_data=numpy.arange(3).tobytes())

>>> db.iquery("load(foo, '{fn}', 0, '(int64)')",
...           upload_data=numpy.arange(3).tobytes())


Upload Data Files
^^^^^^^^^^^^^^^^^

A binary or text file-like object can be used to specify the upload
data. The content of the file has to be in one of the `supported SciDB
formats
<https://paradigm4.atlassian.net/wiki/spaces/ESD169/pages/50856232/input>`_. A
matching format specification has to be provided as well:

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

Please note that the data file is not read into the SciDB-Py
library. The data file object is passed directly to the ``requests``
library which handles the HTTP communication with *Shim*.


SciDB Enterprise Edition Features
=================================

SciDB Enterprise Edition features can be used directly as any other
operators. One special case is the ``set_namespace`` operator. The
operator alters the ``DB`` object on which it is called. The effect
persists until the next call to ``set_namespace``. The operator can be
called directly (i.g., ``db.set_namespace``) or through the ``iquery``
function. No immediate query is executed in SciDB, but the new
namespace will take effect for any subsequent SciDB queries:

>>> print(db)
scidb_url  = http://localhost:8080
scidb_auth = None
http_auth  = None
namespace  = None
verify     = None

Notice the ``namespace`` field of the ``DB`` instance.

>>> db.set_namespace('private')
... # doctest: +SKIP
>>> print(db)
... # doctest: +SKIP
scidb_url  = http://localhost:8080
scidb_auth = None
http_auth  = None
namespace  = private
verify     = None
>>> db.show_namespace()[0]['name']['val']
... # doctest: +SKIP
'private'

>>> db.iquery("set_namespace('public')")
>>> print(db)
scidb_url  = http://localhost:8080
scidb_auth = None
http_auth  = None
namespace  = public
verify     = None
>>> db.show_namespace()[0]['name']['val']
... # doctest: +SKIP
'public'

For convenience, an initial namespace can be provided at connection
time:

>>> db_ssl = connect('https://localhost:8083',
...                  verify=False,
...                  namespace='public')
... # doctest: +SKIP
>>> print(db_ssl)
... # doctest: +SKIP
scidb_url  = https://localhost:8083
scidb_auth = None
http_auth  = None
namespace  = public
verify     = False

"""

from .db import connect, iquery, Array
from .schema import Attribute, Dimension, Schema

__version__ = '16.9.1'
