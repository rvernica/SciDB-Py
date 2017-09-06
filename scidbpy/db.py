"""DB, Array, and Operator
=======================

Classes for connecting to SciDB and executing queries.

"""

import copy
import enum
import itertools
import logging
import numpy
import os
import pandas
import re
import requests
import string
import threading
import warnings

try:
    from weakref import finalize
except ImportError:
    from backports.weakref import finalize

from .meta import ops_hungry, string_args
from .schema import Attribute, Dimension, Schema


class Shim(enum.Enum):
    cancel = 'cancel'
    execute_query = 'execute_query'
    new_session = 'new_session'
    read_bytes = 'read_bytes'
    release_session = 'release_session'
    upload = 'upload'


class Password_Placeholder(object):
    def __repr__(self):
        return 'PASSWORD_PROVIDED'


class DB(object):
    """SciDB Shim connection object.

    >>> DB()
    DB('http://localhost:8080', None, None, None, None)

    >>> print(DB())
    scidb_url  = http://localhost:8080
    scidb_auth = None
    http_auth  = None
    namespace  = None
    verify     = None

    Constructor parameters:

    :param string scidb_url: SciDB connection URL. The URL for the
      Shim server. If `None`, use the value of the `SCIDB_URL`
      environment variable, if present (default
      `http://localhost:8080`)

    :param tuple scidb_auth: Tuple with username and password for
      connecting to SciDB, if password authentication method is used
      (default `None`)

    :param tuple http_auth: Tuple with username and password for
      connecting to Shim, if Shim authentication is used (default
      `None`)

    :param string namespace: Initial namespace for the
      connection. Only applicable for SciDB Enterprise Edition. The
      namespace can changed at any time using the `set_namespace`
      SciDB operator (default `None`)

    :param bool verify: If `False`, HTTPS certificates are not
      verified. This value is passed to the Python `requests`
      library. See Python `requests
      <http://docs.python-requests.org/en/master/>`_ library `SSL Cert
      Verification
      <http://docs.python-requests.org/en/master/user/advanced/
      #ssl-cert-verification>`_ section for details on the ``verify``
      argument (default `None`)

    :param bool no_ops: If `True`, the list of operators is not
      fetched at this time and the connection is not implicitly
      verified. This expedites the execution of the function but
      disallows for calling the SciDB operators directly from the `DB`
      instance e.g., `db.scan` (default `False`)

    """

    _show_query = "show('{}', 'afl')"

    def __init__(
            self,
            scidb_url=None,
            scidb_auth=None,
            http_auth=None,
            namespace=None,
            verify=None,
            no_ops=False):
        if scidb_url is None:
            scidb_url = os.getenv('SCIDB_URL', 'http://localhost:8080')

        self.scidb_url = scidb_url
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

        self._id = None
        self._lock = threading.Lock()
        self._array_cnt = 0
        self._formatter = string.Formatter()

        if no_ops:
            self.operators = None
            self._dir = None
        else:
            self.load_ops()

    def __iter__(self):
        return (i for i in (
            self.scidb_url,
            self.scidb_auth,
            self.http_auth,
            self.namespace,
            self.verify))

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__, *self)

    def __str__(self):
        return '''\
scidb_url  = {}
scidb_auth = {}
http_auth  = {}
namespace  = {}
verify     = {}'''.format(*self)

    def __getattr__(self, name):
        if self.operators and name in self.operators:
            return Operator(self, name)
        else:
            raise AttributeError(
                '{.__name__!r} object has no attribute {!r}'.format(
                    type(self), name))

    def __dir__(self):
        return self._dir

    def iquery(self,
               query,
               fetch=False,
               atts_only=False,
               as_dataframe=False,
               dataframe_promo=True,
               schema=None,
               upload_data=None,
               upload_schema=None):
        """Execute query in SciDB

        :param string query: SciDB AFL query to execute

        :param bool fetch: If `True`, download SciDB array (default
          `False`)

        :param bool atts_only: If `True`, download only SciDB array
          attributes without dimensions (default `False`)

        :param bool as_dataframe: If `True`, return a Pandas
          DataFrame. If `False`, return a NumPy array (default
          `False`)

        :param bool dataframe_promo: If `True`, null-able types are
          promoted as per Pandas 'promotion scheme
          <http://pandas.pydata.org/pandas-docs/stable/gotchas.html
          #na-type-promotions>`_ If `False`, object records are used
          for null-able types (default `True`)

        :param schema: Schema of the SciDB array to use when
          downloading the array. Schema is not verified. If schema is
          a Schema instance, it is copied. Otherwise, a
          :py:class:`Schema` object is built using
          :py:func:`Schema.fromstring` (default `None`)

        >>> DB().iquery('build(<x:int64>[i=0:1; j=0:1], i + j)', fetch=True)
        ... # doctest: +NORMALIZE_WHITESPACE
        array([(0, 0, (255, 0)),
               (0, 1, (255, 1)),
               (1, 0, (255, 1)),
               (1, 1, (255, 2))],
              dtype=[('i', '<i8'), ('j', '<i8'),
                     ('x', [('null', 'u1'), ('val', '<i8')])])

        >>> DB().iquery("input({sch}, '{fn}', 0, '{fmt}')",
        ...             fetch=True,
        ...             upload_data=numpy.arange(3, 6))
        ... # doctest: +NORMALIZE_WHITESPACE
        array([(0, 3), (1, 4), (2, 5)],
              dtype=[('i', '<i8'), ('x', '<i8')])

        """
        # Special case: -- - set_namespace - --
        if query.startswith('set_namespace(') and query[-1] == ')':
            param = query[len('set_namespace('):-1]
            # Unquote if quoted. Will be quoted when set in prefix.
            if param[0] == "'" and param[-1] == "'":
                param = param[1:-1]
            self.namespace = param
            return

        id = self._shim(Shim.new_session).text

        if upload_data is not None:
            if isinstance(upload_data, numpy.ndarray):
                if upload_schema is None:
                    try:
                        upload_schema = Schema.fromdtype(upload_data.dtype)
                    except Exception as e:
                        warnings.warn(
                            'Mapping NumPy dtype to SciDB schema failed. ' +
                            'Try providing an explicit upload_schema')
                        raise e

                # Convert upload data to bytes
                if upload_schema.is_fixsize():
                    upload_data = upload_data.tobytes()
                else:
                    upload_data = upload_schema.tobytes(upload_data)

            # Check if placeholders are present
            place_holders = set(
                field_name
                for _1, field_name, _3, _4 in self._formatter.parse(query))
            if 'fn' not in place_holders:
                warnings.warn(
                    'upload_data provided, but {fn} placeholder is missing',
                    stacklevel=2)
            if 'fmt' in place_holders and upload_schema is None:
                warnings.warn(
                    'upload_data and {fmt} placeholder provided, ' +
                    'but upload_schema is None',
                    stacklevel=2)

            # Check if upload data is bytes or file-like object
            if not (isinstance(upload_data, bytes) or
                    isinstance(upload_data, bytearray) or
                    hasattr(upload_data, 'read')):
                print('data type')
                warnings.warn(
                    'upload_data is not bytes or file-like object',
                    stacklevel=2)

            fn = self._shim(Shim.upload, id=id, data=upload_data).text
            query = query.format(
                sch=upload_schema,
                fn=fn,
                fmt=upload_schema.atts_fmt_scidb if upload_schema else None)

        if fetch:
            # Use provided schema or get schema from SciDB
            if schema:
                # Deep-copy schema since we might be mutating it
                if isinstance(schema, Schema):
                    if not atts_only:
                        schema = copy.deepcopy(schema)
                else:
                    schema = Schema.fromstring(schema)
            else:
                # Execute 'show(...)' and Download text
                self._shim(
                    Shim.execute_query,
                    id=id,
                    query=DB._show_query.format(query.replace("'", "\\'")),
                    save='tsv')
                schema = Schema.fromstring(
                    self._shim(Shim.read_bytes, id=id, n=0).text)

            # Attributes and dimensions can collide. Run make_unique to
            # remove any collisions.
            #
            # make_unique fixes any collision, but if we don't
            # download the dimensions, we don't need to fix collisions
            # between dimensions and attributes. So, we use
            # make_unique only if there are collisions within the
            # attribute names.
            if ((not atts_only or
                 len(set((a.name for a in schema.atts))) <
                 len(schema.atts)) and schema.make_unique()):
                # Dimensions or attributes were renamed due to
                # collisions. We need to cast.
                query = 'cast({}, {:h})'.format(query, schema)

            # Unpack
            if not atts_only:
                # apply: add dimensions as attributes
                # project: place dimensions first
                query = 'project(apply({}, {}), {})'.format(
                    query,
                    ', '.join('{0}, {0}'.format(d.name) for d in schema.dims),
                    ', '.join(i.name for i in itertools.chain(
                        schema.dims, schema.atts)))

                # update schema after apply
                schema.make_dims_atts()

            # Execute Query and Download content
            self._shim(Shim.execute_query,
                       id=id,
                       query=query,
                       save=schema.atts_fmt_scidb)
            buf = self._shim(Shim.read_bytes, id=id, n=0).content

            self._shim(Shim.release_session, id=id)

            if schema.is_fixsize() and (not as_dataframe or
                                        not dataframe_promo):
                data = numpy.frombuffer(buf, dtype=schema.atts_dtype)
            else:
                data = schema.frombytes(buf, as_dataframe, dataframe_promo)

            # Return NumPy array or Pandas dataframe
            if as_dataframe:
                return pandas.DataFrame.from_records(data)
            else:
                return data

        else:                   # fetch=False
            self._shim(Shim.execute_query, id=id, query=query, release=1)

            # Special case: -- - load_library - --
            if query.startswith('load_library('):
                self.load_ops()

    def iquery_readlines(self, query):
        """Execute query in SciDB

        >>> DB().iquery_readlines('build(<x:int64>[i=0:2], i * i)')
        ... # doctest: +ELLIPSIS
        [...'0', ...'1', ...'4']

        >>> DB().iquery_readlines(
        ...   'apply(build(<x:int64>[i=0:2], i), y, i + 10)')
        ... # doctest: +ELLIPSIS
        [[...'0', ...'10'], [...'1', ...'11'], [...'2', ...'12']]
        """
        id = self._shim(Shim.new_session).text
        self._shim(Shim.execute_query, id=id, query=query, save='tsv')
        ret = self._shim_readlines(id=id)
        self._shim(Shim.release_session, id=id)
        return ret

    def next_array_name(self):
        """Generate a uniqu array name. Keep track on these names using the
           _id field and a conter
        """
        # Thread-safe counter
        with self._lock:
            self._array_cnt += 1
            return 'py_{}_{}'.format(self._id, self._array_cnt)

    def load_ops(self):
        """Get list of operators and macros. Also sets the _id field used to
           generate unique array names
        """
        id = self._shim(Shim.new_session).text

        query_id = self._shim(
            Shim.execute_query,
            id=id,
            query="project(list('operators'), name)",
            save='tsv').text  # set query ID as DB instance ID
        if self._id is None:
            self._id = query_id
        operators = self._shim_readlines(id=id)

        self._shim(
            Shim.execute_query,
            id=id,
            query="project(list('macros'), name)",
            save='tsv').content
        macros = self._shim_readlines(id=id)

        self._shim(Shim.release_session, id=id)

        self.operators = operators + macros
        self._dir = (self.operators +
                     ['arrays',
                      'gc',
                      'iquery',
                      'iquery_readlines',
                      'upload'])
        self._dir.sort()

    def _shim(self, endpoint, **kwargs):
        """Make request on Shim endpoint"""

        # Add credentails to request, if necessary
        if self._scidb_auth and endpoint in (Shim.cancel, Shim.execute_query):
            kwargs.update(self._scidb_auth)

        # Add prefix to request, if necessary
        if self.namespace and endpoint == Shim.execute_query:
            kwargs['prefix'] = "set_namespace('{}')".format(self.namespace)

        # Make request
        url = requests.compat.urljoin(self.scidb_url, endpoint.value)
        if endpoint == Shim.upload:  # Post request
            req = requests.post(
                '{}?id={}'.format(url, kwargs['id']),
                data=kwargs['data'],
                auth=self._http_auth,
                verify=self.verify)
        else:                        # Get request
            req = requests.get(
                url,
                params=kwargs,
                auth=self._http_auth,
                verify=self.verify)
        req.reason = req.content
        req.raise_for_status()
        return req

    def _shim_readlines(self, id):
        """Read data from Shim and parse as text lines"""
        return [line.split('\t') if '\t' in line else line
                for line in self._shim(
                        Shim.read_bytes, id=id, n=0).text.splitlines()]


class Arrays(object):
    """Access to arrays available in SciDB"""
    def __init__(self, db):
        self.db = db

    def __repr__(self):
        return '{}({!r})'.format(
            type(self).__name__, self.db)

    def __str__(self):
        return '''DB:
{}'''.format(self.db)

    def __getattr__(self, name):
        return Array(self.db, name)

    def __dir__(self):
        """Download the list of SciDB arrays. Use 'project(list(), name)' to
        download only names and schemas
        """
        return self.db.iquery_readlines('project(list(), name)')


class Array(object):
    """Access to individual array"""
    def __init__(self, db, name, gc=False):
        self.db = db
        self.name = name

        if gc:
            finalize(self,
                     self.db.iquery,
                     'remove({})'.format(self.name))

    def __repr__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.db, self.name)

    def __str__(self):
        return self.name

    def __getattr__(self, key):
        return ArrayExp('{}.{}'.format(self.name, key))

    def __getitem__(self, key):
        return self.fetch()[key]

    def __dir__(self):
        """Download the schema of the SciDB array, using `show()`"""
        sh = Schema.fromstring(
            self.db.iquery_readlines('show({})'.format(self))[0])
        ls = [i.name for i in itertools.chain(sh.atts, sh.dims)]
        ls.sort()
        return ls

    def fetch(self, as_dataframe=False):
        return self.db.iquery(
            'scan({})'.format(self), fetch=True, as_dataframe=as_dataframe)


class ArrayExp(object):
    """Access to individual attribute or dimension"""
    def __init__(self, exp):
        self.exp = exp

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.exp)

    def __str__(self):
        return '{}'.format(self.exp)

    def __add__(self, other):
        return ArrayExp('{} + {}'.format(self, other))


class Operator(object):
    """Store SciDB operator and arguments. Hungry operators (e.g., remove,
    store, etc.) evaluate immediately. Lazy operators evaluate on data
    fetch.

    """
    def __init__(self, db, name, upload_data=None, upload_schema=None, *args):
        self.db = db
        self.name = name.lower()
        self.upload_data = upload_data
        self.upload_schema = upload_schema

        self.args = list(args)
        self.is_lazy = self.name not in ops_hungry

        self._dir = self.db.operators + ['fetch']
        self._dir.sort()

    def __repr__(self):
        return '{}(db={!r}, name={!r}, args=[{}])'.format(
            type(self).__name__,
            self.db,
            self.name,
            ', '.join('{!r}'.format(i) for i in self.args))

    def __str__(self):
        args_fmt = []
        for (pos, arg) in enumerate(self.args):
            # Format argument to string (possibly recursive)
            arg_fmt = '{}'.format(arg)

            # Special case: quote string argument if not quoted
            if (self.name in string_args[pos] and
                    arg and
                    arg_fmt[0] != "'" and
                    arg_fmt[-1] != "'"):

                arg_fmt = "'{}'".format(arg_fmt)

            # Add to arguments list
            args_fmt.append(arg_fmt)

        return '{}({})'.format(self.name, ', '.join(args_fmt))

    def __call__(self, *args, **kwargs):
        """Returns self for lazy expressions. Executes immediate expressions.
        """
        self.args.extend(args)

        # Special case: -- - create_array - --
        if self.name == 'create_array' and len(self.args) < 3:
            # Set "temporary"
            self.args.append(False)

        # Special case: -- - input & load - --
        elif self.name in ('input', 'load'):
            ln = len(self.args)

            # Set upload data
            if 'upload_data' in kwargs.keys():
                self.upload_data = kwargs['upload_data']
            # Set upload schema
            if 'upload_schema' in kwargs.keys():
                # Pass through if provided as argument
                self.upload_schema = kwargs['upload_schema']
            if self.upload_schema is None:
                # If the upload_data is a NumPy array try to map the
                # array dtype to upload schema
                if (self.upload_data is not None and
                        isinstance(self.upload_data, numpy.ndarray)):
                    try:
                        self.upload_schema = Schema.fromdtype(
                            self.upload_data.dtype)
                    except:
                        # Might fail if the dtype contains
                        # objects. The same type mapping is attempted
                        # later in iquery, but there the exception is
                        # propagated
                        pass
                # If the oprator is input, try to map first argument
                # to upload schema
                if (self.upload_schema is None and
                        self.name == 'input' and
                        ln >= 1):
                    try:
                        self.upload_schema = Schema.fromstring(args[0])
                    except:
                        # Fails if the argument is an array name
                        pass

            # Set defaults if arguments are missing
            # Check if "format" is present (4th argument)
            if ln < 4:
                # Check if "instance_id" is present (3rd argument)
                if ln < 3:
                    # Check if "input_file" is present (2nd argument)
                    if ln < 2:
                        # Check if "existing_array|anonymous_schema"
                        # is present (1st argument)
                        if ln < 1:
                            self.args.append('{sch}')  # anonymous_schema
                        self.args.append("'{fn}'")     # input_file
                    self.args.append(0)                # instance_id
                self.args.append("'{fmt}'")            # format

        # Special case: -- - store - --
        elif self.name == 'store' and len(self.args) < 2:
            # Set "named_array"
            self.args.append(self.db.next_array_name())
            # Garbage collect (if not specified)
            if 'gc' not in kwargs.keys():
                kwargs['gc'] = True

        # Lazy or hungry
        if self.is_lazy:        # Lazy
            return self

        else:                   # Hungry
            # Execute query
            self.db.iquery(str(self),
                           upload_data=self.upload_data,
                           upload_schema=self.upload_schema)

            # Handle output
            # Special case: -- - load - --
            if self.name == 'load':
                if isinstance(self.args[0], Array):
                    return self.args[0]
                else:
                    return Array(self.db, self.args[0])

            # Special case: -- - store - --
            elif self.name == 'store':
                if isinstance(self.args[1], Array):
                    return self.args[1]
                else:
                    return Array(self.db,
                                 self.args[1],
                                 kwargs.get('gc', False))

    def __getitem__(self, key):
        return self.fetch()[key]

    def __getattr__(self, name):
        if name in self.db.operators:
            return Operator(
                self.db, name, self.upload_data, self.upload_schema, self)
        else:
            raise AttributeError(
                '{.__name__!r} object has no attribute {!r}'.format(
                    type(self), name))

    def __dir__(self):
        return self._dir

    def fetch(self, atts_only=False, as_dataframe=False):
        if self.is_lazy:
            return self.db.iquery(str(self),
                                  fetch=True,
                                  atts_only=atts_only,
                                  as_dataframe=as_dataframe,
                                  upload_data=self.upload_data,
                                  upload_schema=self.upload_schema)


connect = DB
iquery = DB.iquery


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    import doctest
    doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)
