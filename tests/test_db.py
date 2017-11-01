import numpy
import pytest
import random

from scidbpy.db import Array, connect, iquery
from scidbpy.schema import Schema


@pytest.fixture(scope='module')
def db():
    return connect()


class TestDB:

    @pytest.mark.parametrize('args', [
        {'scidb_auth': ('foo', 'bar')},
        {'scidb_url': 'http://localhost:8080', 'scidb_auth': ('foo', 'bar')},
    ])
    def test_connect_exception(self, args):
        with pytest.raises(Exception):
            connect(**args)

    @pytest.mark.parametrize('query', [
        'list()',
        "list('operators')",
    ])
    def test_iquery(self, db, query):
        assert db.iquery(query) is None
        assert type(
            db.iquery(query, fetch=True, as_dataframe=False)) == numpy.ndarray

    @pytest.mark.parametrize('query_batch', [
        [('store(build(<val:double>[i=1:10,10,0], i), foo)', False),
         ('versions(foo)', True),
         ('remove(foo)', False)],
    ])
    def test_iquery_batch(self, db, query_batch):
        for (query, fetch) in query_batch:
            if fetch:
                assert type(db.iquery(
                    query, fetch=fetch, as_dataframe=False)) == numpy.ndarray
            else:
                assert db.iquery(
                    query, fetch=fetch, as_dataframe=False) is None

    @pytest.mark.parametrize(('type_name', 'schema'), [
        (type_name, schema)
        for type_name in [
                '{}int{}'.format(pre, sz)
                for pre in ('', 'u')
                for sz in (8, 16, 32, 64)
        ] + [
            'bool',
            'double',
            'float',
        ] + [
            '{}int{}'.format(u, sz)
            for u in ('', 'u') for sz in (8, 16, 32, 64)
        ]
        for schema in [
                None,
                'build<val:{}>[i=1:10,10,0]'.format(type_name),
                '<val:{}>[i=1:10,10,0]'.format(type_name),
                '<val:{}>[i=1:10]'.format(type_name),
                '<val:{}>[i]'.format(type_name),
        ]
    ])
    @pytest.mark.parametrize('atts_only', [
        True,
        False,
    ])
    def test_fetch_numpy(self, db, type_name, atts_only, schema):
        # NumPy array
        ar = iquery(
            db,
            'build(<val:{}>[i=1:10,10,0], random())'.format(type_name),
            fetch=True,
            atts_only=atts_only,
            as_dataframe=False,
            schema=schema)
        if not atts_only:
            for i in range(10):
                assert ar[i][0] == i + 1
        assert ar.shape == (10,)
        assert ar.ndim == 1

    @pytest.mark.parametrize(('type_name', 'schema'), [
        (type_name, schema)
        for type_name in [
                '{}int{}'.format(pre, sz)
                for pre in ('', 'u')
                for sz in (8, 16, 32, 64)
        ] + [
            'bool',
            'double',
            'float',
            'int8',
            'uint64',
        ]
        for schema in [
                None,
                'build<val:{}>[i=1:10,10,0]'.format(type_name),
                '<val:{}>[i=1:10,10,0]'.format(type_name),
                '<val:{}>[i=1:10]'.format(type_name),
                '<val:{}>[i]'.format(type_name),
        ]
    ])
    def test_fetch_dataframe(self, db, type_name, schema):
        # Pandas DataFrame
        ar = iquery(
            db,
            'build(<val:{}>[i=1:10,10,0], random())'.format(type_name),
            fetch=True,
            schema=schema)
        assert ar.shape == (10, 2)
        assert ar.ndim == 2

    @pytest.mark.parametrize(('type_name', 'schema'), [
        (type_name, schema)
        for type_name in [
                '{}int{}'.format(pre, sz)
                for pre in ('', 'u')
                for sz in (8, 16, 32, 64)
        ] + [
            'bool',
            'double',
            'float',
            'int8',
            'uint64',
        ]
        for schema in [
                None,
                'build<val:{}>[i=1:10,10,0]'.format(type_name),
                '<val:{}>[i=1:10,10,0]'.format(type_name),
                '<val:{}>[i=1:10]'.format(type_name),
                '<val:{}>[i]'.format(type_name),
        ]
    ])
    def test_fetch_dataframe_atts(self, db, type_name, schema):
        # Pandas DataFrame
        ar = iquery(
            db,
            'build(<val:{}>[i=1:10,10,0], random())'.format(type_name),
            fetch=True,
            atts_only=True,
            schema=schema)
        assert ar.shape == (10, 1)
        assert ar.ndim == 2


variety_atts_types = """b  :bool       null,
                        c  :char       null,
                        d  :double     null,
                        f  :float      null,
                        i8 :int8       null,
                        i16:int16      null,
                        i32:int32      null,
                        i64:int64      null,
                        s  :string     null,
                        u8 :uint8      null,
                        u16:uint16     null,
                        u32:uint32     null,
                        u64:uint64     null"""

variety_atts = [at.split(':')[0].strip()
                for at in variety_atts_types.split(',')]

variety_queries = {
    'create': [
        'create array variety_i<{}>[i=0:2]'.format(variety_atts_types),

        'create array variety<{}>[i=0:2; j=-2:0; k=0:*]'.format(
            variety_atts_types),

        """store(
             build(
               variety_i,
               '[(true,
                  "a",
                  1.7976931348623157e+308,
                  -3.4028234663852886e+38,
                  -128,
                  -32768,
                  -2147483648,
                  -9223372036854775808,
                  "abcDEF123",
                  255,
                  65535,
                  4294967295,
                  18446744073709551615),
                 (null,
                  null,
                  null,
                  ?34,
                  ?7,
                  ?15,
                  ?31,
                  ?63,
                  null,
                  null,
                  null,
                  null,
                  null),
                 (?99,
                  ?65,
                  -inf,
                  inf,
                  null,
                  null,
                  null,
                  null,
                  ?99,
                  ?8,
                  ?16,
                  ?32,
                  ?64)]',
               true),
             variety_i)""",

        """store(
             filter(
               redimension(
                 cross_join(
                   cross_join(
                     variety_i,
                     build(<x:int8 null>[j=-2:0], null)),
                   build(<y:int8 null>[k=0:2], null)),
                 variety),
               j != 0 and k != 1),
             variety)"""],
    'clean': [
        'remove(variety_i)',
        'remove(variety)']}

variety_schema = variety_queries['create'][1][len('create array '):]

variety_array_struct = numpy.array(
    [(0, -2, 0,
      (255, True),
      (255, b'a'),
      (255, 1.7976931348623157e+308),
      (255, -3.4028234663852886e+38),
      (255, -128),
      (255, -32768),
      (255, -2147483648),
      (255, -9223372036854775808),
      (255, 'abcDEF123'),
      (255, 255),
      (255, 65535),
      (255, 4294967295),
      (255, 18446744073709551615)),
     (1, -2, 0,
      (0, False),
      (0, b''),
      (0, 0.0),
      (34, 0.0),
      (7, 0),
      (15, 0),
      (31, 0),
      (63, 0),
      (0, ''),
      (0, 0),
      (0, 0),
      (0, 0),
      (0, 0)),
     (2, -2, 0,
      (99, False),
      (65, b''),
      (255, -numpy.inf),
      (255, numpy.inf),
      (0, 0),
      (0, 0),
      (0, 0),
      (0, 0),
      (99, ''),
      (8, 0),
      (16, 0),
      (32, 0),
      (64, 0))],
    dtype=[('i', '<i8'), ('j', '<i8'), ('k', '<i8'),
           ('b', [('null', 'u1'), ('val', '?')]),
           ('c', [('null', 'u1'), ('val', 'S1')]),
           ('d', [('null', 'u1'), ('val', '<f8')]),
           ('f', [('null', 'u1'), ('val', '<f4')]),
           ('i8', [('null', 'u1'), ('val', 'i1')]),
           ('i16', [('null', 'u1'), ('val', '<i2')]),
           ('i32', [('null', 'u1'), ('val', '<i4')]),
           ('i64', [('null', 'u1'), ('val', '<i8')]),
           ('s', [('null', 'u1'), ('val', 'O')]),
           ('u8', [('null', 'u1'), ('val', 'u1')]),
           ('u16', [('null', 'u1'), ('val', '<u2')]),
           ('u32', [('null', 'u1'), ('val', '<u4')]),
           ('u64', [('null', 'u1'), ('val', '<u8')])])

variety_array_obj = variety_array_struct.astype(
    [(d, '<i8') for d in ('i', 'j', 'k')] +
    [(a, 'O') for a in variety_atts])

variety_array_promo = numpy.array(
    [[0, -2, 0,
      True,
      b'a',
      1.7976931348623157e+308,
      -3.4028234663852886e+38,
      -128.0,
      -32768.0,
      -2147483648.0,
      -9.223372036854776e+18,
      'abcDEF123',
      255.0,
      65535.0,
      4294967295.0,
      1.8446744073709552e+19],
     [1, -2, 0,
      None,
      None,
      numpy.nan,
      numpy.nan,
      numpy.nan,
      numpy.nan,
      numpy.nan,
      numpy.nan,
      None,
      numpy.nan,
      numpy.nan,
      numpy.nan,
      numpy.nan],
     [2, -2, 0,
      None,
      None,
      -numpy.inf,
      numpy.inf,
      numpy.nan,
      numpy.nan,
      numpy.nan,
      numpy.nan,
      None,
      numpy.nan,
      numpy.nan,
      numpy.nan,
      numpy.nan]], dtype=object)


@pytest.fixture(scope='module')
def variety(db):
    for q in variety_queries['create']:
        iquery(db, q)
    variety = variety_schema.split('<')[0].strip()

    yield variety
    for q in variety_queries['clean']:
        iquery(db, q)


class TestVariety:

    @pytest.mark.parametrize('schema', [
        None,
        variety_schema,
        Schema.fromstring(variety_schema),
    ])
    def test_variety_numpy(self, db, variety, schema):
        # NumPy array
        ar = iquery(db,
                    'scan({})'.format(variety),
                    fetch=True,
                    as_dataframe=False,
                    schema=schema)
        assert ar.shape == (12,)
        assert ar.ndim == 1
        assert ar[0] == variety_array_struct[0]
        assert ar[4] == variety_array_struct[1]
        assert ar[8] == variety_array_struct[2]

    @pytest.mark.parametrize('schema', [
        None,
        variety_schema,
        Schema.fromstring(variety_schema),
    ])
    def test_variety_numpy_atts(self, db, variety, schema):
        # NumPy array, atts_only
        ar = iquery(db,
                    'scan({})'.format(variety),
                    fetch=True,
                    atts_only=True,
                    as_dataframe=False,
                    schema=schema)
        assert ar.shape == (12,)
        assert ar.ndim == 1
        assert ar[0] == variety_array_struct[variety_atts][0]
        assert ar[4] == variety_array_struct[variety_atts][1]
        assert ar[8] == variety_array_struct[variety_atts][2]

    @pytest.mark.parametrize('schema', [
        None,
        variety_schema,
        Schema.fromstring(variety_schema),
    ])
    def test_variety_dataframe(self, db, variety, schema):
        # Pandas DataFrame, atts_only
        ar = iquery(db,
                    'scan({})'.format(variety),
                    fetch=True)
        assert ar.shape == (12, 16)
        assert ar.ndim == 2
        assert numpy.all(ar[0:1].values[0] == variety_array_promo[0])

        # Values which differ have to be NAN
        ln = ar[4:5]
        assert numpy.all(
            numpy.isnan(
                ln[ar.columns[
                    (ln.values[0] != variety_array_promo[1]).tolist()]]))

        ln = ar[8:9]
        assert numpy.all(
            numpy.isnan(
                ln[ar.columns[
                    (ln.values[0] != variety_array_promo[2]).tolist()]]))

    @pytest.mark.parametrize('schema', [
        None,
        variety_schema,
        Schema.fromstring(variety_schema),
    ])
    def test_variety_dataframe_atts(self, db, variety, schema):
        # Pandas DataFrame, atts_only
        ar = iquery(db,
                    'scan({})'.format(variety),
                    fetch=True,
                    atts_only=True)
        assert ar.shape == (12, 13)
        assert ar.ndim == 2
        assert numpy.all(ar[0:1].values == variety_array_promo[0][3:])

        # Values which differ have to be NAN
        ln = ar[4:5]
        assert numpy.all(
            numpy.isnan(
                ln[ar.columns[(ln.values != variety_array_promo[1][3:])[0]]]))

        ln = ar[8:9]
        assert numpy.all(
            numpy.isnan(
                ln[ar.columns[(ln.values != variety_array_promo[2][3:])[0]]]))

    @pytest.mark.parametrize('schema', [
        None,
        variety_schema,
        Schema.fromstring(variety_schema),
    ])
    def test_variety_dataframe_no_promo(self, db, variety, schema):
        # Pandas DataFrame, atts_only
        ar = iquery(db,
                    'scan({})'.format(variety),
                    fetch=True,
                    dataframe_promo=False)
        assert ar.shape == (12, 16)
        assert ar.ndim == 2
        assert ar[0:1].to_records(index=False) == variety_array_obj[0]
        assert ar[4:5].to_records(index=False) == variety_array_obj[1]
        assert ar[8:9].to_records(index=False) == variety_array_obj[2]


foo_np = numpy.random.randint(1e3, size=10)
foo_np_null = numpy.array([((255, random.randint(0, 1e3)),)
                           for _ in range(10)],
                          dtype=[('val', [('null', 'u1'), ('val', '<i8')])])


class TestUpload:

    # -- - --
    # -- - IQuery - --
    # -- - --
    @pytest.mark.parametrize(('query', 'upload_schema_str'), [
        (pre + "(foo, '{fn}', 0, '" + fmt + "'" + suf + ')', ups)
        for fmt in ('(int64)', '{fmt}')
        for (pre, suf) in (('store(input', '), foo'),
                           ('insert(input', '), foo'),
                           ('load', ''))
        for ups in ('<val:int64 not null>[i]', None)
    ] + [
        (pre + '(' +
         'input(' + sch + ", '{fn}', 0, '" + fmt + "'), " +
         'foo)', ups)
        for fmt in ('(int64)', '{fmt}')
        for sch in ('<val:int64>[i]', '<val:int64 not null>[i]', '{sch}')
        for pre in ('store',
                    'insert')
        for ups in ('<val:int64 not null>[i]', None)
    ])
    def test_iquery_numpy(self, db, query, upload_schema_str):
        db.create_array('foo', '<val:int64>[i]')
        assert db.iquery(
            query,
            upload_data=foo_np,
            upload_schema=(Schema.fromstring(upload_schema_str)
                           if upload_schema_str else None)) is None
        db.remove('foo')

    @pytest.mark.parametrize(('query', 'upload_schema_str'), [
        (pre + "(foo, '{fn}', 0, '" + fmt + "'" + suf + ')', ups)
        for fmt in ('(int64 null)', '{fmt}')
        for (pre, suf) in (('store(input', '), foo'),
                           ('insert(input', '), foo'),
                           ('load', ''))
        for ups in ('<val:int64>[i]', None)
    ] + [
        (pre + '(' +
         'input(' + sch + ", '{fn}', 0, '" + fmt + "'), " +
         'foo)', ups)
        for fmt in ('(int64 null)', '{fmt}')
        for sch in ('<val:int64>[i]', '<val:int64 not null>[i]', '{sch}')
        for pre in ('store', 'insert')
        for ups in ('<val:int64>[i]', None)
    ])
    def test_iquery_numpy_null(self, db, query, upload_schema_str):
        db.create_array('foo', '<val:int64>[i]')
        assert db.iquery(
            query,
            upload_data=foo_np_null,
            upload_schema=(Schema.fromstring(upload_schema_str)
                           if upload_schema_str else None)) is None
        db.remove('foo')

    @pytest.mark.parametrize(('query', 'upload_schema_str'), [
        (pre + "(foo, '{fn}', 0, '" + fmt + "'" + suf + ')', ups)
        for fmt in ('(int64)', '{fmt}')
        for (pre, suf) in (('store(input', '), foo'),
                           ('insert(input', '), foo'),
                           ('load', ''))
        for ups in (['<val:int64 not null>[i]'] +
                    ([None] if fmt[0] != '{' else []))
    ] + [
        ((pre + '(' +
          'input(' + sch + ", '{fn}', 0, '" + fmt + "'), " +
          'foo)'), ups)
        for fmt in ('(int64)', '{fmt}')
        for sch in ('<val:int64>[i]', '<val:int64 not null>[i]', '{sch}')
        for pre in ('store', 'insert')
        for ups in (['<val:int64 not null>[i]'] +
                    ([None] if fmt[0] != '{' and sch[0] != '{' else []))
    ])
    def test_iquery_numpy_bytes(self, db, query, upload_schema_str):
        db.create_array('foo', '<val:int64>[i]')
        assert db.iquery(
            query,
            upload_data=foo_np,
            upload_schema=(Schema.fromstring(upload_schema_str)
                           if upload_schema_str else None)) is None
        db.remove('foo')

    @pytest.mark.parametrize(('query', 'upload_schema_str'), [
        (pre + "(foo, '{fn}', 0, '" + fmt + "'" + suf + ')',
         ups)
        for fmt in ('(int64 null)', '{fmt}')
        for (pre, suf) in (('store(input', '), foo'),
                           ('insert(input', '), foo'),
                           ('load', ''))
        for ups in (['<val:int64>[i]'] +
                    ([None] if fmt[0] != '{' else []))
    ] + [
        (pre + '(' +
         'input(' + sch + ", '{fn}', 0, '" + fmt + "'), " +
         'foo)',
         ups)
        for fmt in ('(int64 null)', '{fmt}')
        for sch in ('<val:int64>[i]', '<val:int64 not null>[i]', '{sch}')
        for pre in ('store', 'insert')
        for ups in (['<val:int64>[i]'] +
                    ([None] if fmt[0] != '{' and sch[0] != '{' else []))
    ])
    def test_iquery_numpy_null_bytes(self, db, query, upload_schema_str):
        db.create_array('foo', '<val:int64>[i]')
        assert db.iquery(
            query,
            upload_data=foo_np_null.tobytes(),
            upload_schema=(Schema.fromstring(upload_schema_str)
                           if upload_schema_str else None)) is None
        db.remove('foo')

    # -- - --
    # -- - Input - --
    # -- - --
    @pytest.mark.parametrize(('args', 'upload_schema_str'), [
        ((arr, inp, ins, fmt) if all((arr, inp, ins, fmt)) else
         (arr, inp, ins) if all((arr, inp, ins)) else
         (arr, inp) if arr and inp else
         (arr,) if arr else
         (), ups)
        for arr in ('',
                    'foo',
                    'foo_not_null',
                    '<val:int64>[i]',
                    '<val:int64 not null>[i]',
                    '{sch}')
        for inp in (('', "'{fn}'") if arr else (None,))
        for ins in (('', '0') if inp else (None,))
        for fmt in (('', "'(int64)'", "'{fmt}'") if ins else (None,))
        for ups in ('<val:int64 not null>[i]', None)
    ])
    def test_input_numpy(self, db, args, upload_schema_str):
        if args and args[0].startswith('foo'):
            if args[0].endswith('not_null'):
                args = ('foo',) + args[1:]
                db.create_array('foo', '<val:int64 not null>[i]')
            else:
                db.create_array('foo', '<val:int64>[i]')
        assert type(
            db.input(*args,
                     upload_data=foo_np,
                     upload_schema=(Schema.fromstring(upload_schema_str)
                                    if upload_schema_str else None)).store(
                                            'foo')) == Array
        db.remove('foo')

    @pytest.mark.parametrize(('args', 'upload_schema_str'), [
        ((arr, inp, ins, fmt) if all((arr, inp, ins, fmt)) else
         (arr, inp, ins) if all((arr, inp, ins)) else
         (arr, inp) if arr and inp else
         (arr,) if arr else
         (), ups)
        for arr in ('',
                    'foo',
                    'foo_not_null',
                    '<val:int64>[i]',
                    '<val:int64 not null>[i]',
                    '{sch}')
        for inp in (('', "'{fn}'") if arr else (None,))
        for ins in (('', '0') if inp else (None,))
        for fmt in (('', "'(int64 null)'", "'{fmt}'") if ins else (None,))
        for ups in ('<val:int64>[i]', None)
    ])
    def test_input_numpy_null(self, db, args, upload_schema_str):
        if args and args[0].startswith('foo'):
            if args[0].endswith('not_null'):
                args = ('foo',) + args[1:]
                db.create_array('foo', '<val:int64 not null>[i]')
            else:
                db.create_array('foo', '<val:int64>[i]')
        assert type(
            db.input(*args,
                     upload_data=foo_np_null,
                     upload_schema=(Schema.fromstring(upload_schema_str)
                                    if upload_schema_str else None)).store(
                                            'foo')) == Array
        db.remove('foo')

    @pytest.mark.parametrize(('args', 'upload_schema_str'), [
        ((arr, inp, ins, fmt) if all((arr, inp, ins, fmt)) else
         (arr, inp, ins) if all((arr, inp, ins)) else
         (arr, inp) if arr and inp else
         (arr,) if arr else
         (), ups)
        for arr in ('',
                    'foo',
                    'foo_not_null',
                    '<val:int64>[i]',
                    '<val:int64 not null>[i]',
                    '{sch}')
        for inp in (('', "'{fn}'") if arr else (None,))
        for ins in (('', '0') if inp else (None,))
        for fmt in (('', "'(int64)'", "'{fmt}'") if ins else (None,))
        for ups in (['<val:int64 not null>[i]'] +
                    ([None] if (arr == '<val:int64 not null>[i]' or
                                (arr not in ('', '{sch}') and
                                 fmt == "'(int64)'")) else []))
    ])
    def test_input_numpy_bytes(self, db, args, upload_schema_str):
        if args and args[0].startswith('foo'):
            if args[0].endswith('not_null'):
                args = ('foo',) + args[1:]
                db.create_array('foo', '<val:int64 not null>[i]')
            else:
                db.create_array('foo', '<val:int64>[i]')
        assert type(
            db.input(*args,
                     upload_data=foo_np.tobytes(),
                     upload_schema=(Schema.fromstring(upload_schema_str)
                                    if upload_schema_str else None)).store(
                                            'foo')) == Array
        db.remove('foo')

    @pytest.mark.parametrize(('args', 'upload_schema_str'), [
        ((arr, inp, ins, fmt) if all((arr, inp, ins, fmt)) else
         (arr, inp, ins) if all((arr, inp, ins)) else
         (arr, inp) if arr and inp else
         (arr,) if arr else
         (), ups)
        for arr in ('',
                    'foo',
                    'foo_not_null',
                    '<val:int64>[i]',
                    '<val:int64 not null>[i]',
                    '{sch}')
        for inp in (('', "'{fn}'") if arr else (None,))
        for ins in (('', '0') if inp else (None,))
        for fmt in (('', "'(int64 null)'", "'{fmt}'") if ins else (None,))
        for ups in (['<val:int64>[i]'] +
                    ([None] if (arr == '<val:int64>[i]' or
                                (arr not in ('', '{sch}') and
                                 fmt == "'(int64 null)'")) else []))
    ])
    def test_input_numpy_null_bytes(self, db, args, upload_schema_str):
        if args and args[0].startswith('foo'):
            if args[0].endswith('not_null'):
                args = ('foo',) + args[1:]
                db.create_array('foo', '<val:int64 not null>[i]')
            else:
                db.create_array('foo', '<val:int64>[i]')
        assert type(
            db.input(*args,
                     upload_data=foo_np_null.tobytes(),
                     upload_schema=(Schema.fromstring(upload_schema_str)
                                    if upload_schema_str else None)).store(
                                            'foo')) == Array
        db.remove('foo')

    # -- - --
    # -- - Load - --
    # -- - --
    @pytest.mark.parametrize(('args', 'upload_schema_str'), [
        ((arr, inp, ins, fmt) if all((inp, ins, fmt)) else
         (arr, inp, ins) if inp and ins else
         (arr, inp) if inp else
         (arr,), ups)
        for arr in ('foo', 'foo_not_null')
        for inp in ('', "'{fn}'")
        for ins in (('', '0') if inp else (None,))
        for fmt in (('', "'(int64)'", "'{fmt}'") if ins else (None,))
        for ups in ('<val:int64 not null>[i]', None)
    ])
    def test_load_numpy(self, db, args, upload_schema_str):
        if args and args[0].startswith('foo'):
            if args[0].endswith('not_null'):
                args = ('foo',) + args[1:]
                db.create_array('foo', '<val:int64 not null>[i]')
            else:
                db.create_array('foo', '<val:int64>[i]')
        assert type(
            db.load('foo',
                    *args[1:],
                    upload_data=foo_np,
                    upload_schema=(Schema.fromstring(upload_schema_str)
                                   if upload_schema_str else None))) == Array
        db.remove('foo')

    @pytest.mark.parametrize(('args', 'upload_schema_str'), [
        ((arr, inp, ins, fmt) if all((inp, ins, fmt)) else
         (arr, inp, ins) if inp and ins else
         (arr, inp) if inp else
         (arr,), ups)
        for arr in ('foo', 'foo_not_null')
        for inp in ('', "'{fn}'")
        for ins in (('', '0') if inp else (None,))
        for fmt in (('', "'(int64 null)'", "'{fmt}'") if ins else (None,))
        for ups in ('<val:int64>[i]', None)
    ])
    def test_load_numpy_null(self, db, args, upload_schema_str):
        if args[0].endswith('not_null'):
            args = ('foo',) + args[1:]
            db.create_array('foo', '<val:int64 not null>[i]')
        else:
            db.create_array('foo', '<val:int64>[i]')
        assert type(
            db.load('foo',
                    *args[1:],
                    upload_data=foo_np_null,
                    upload_schema=(Schema.fromstring(upload_schema_str)
                                   if upload_schema_str else None))) == Array
        db.remove('foo')

    @pytest.mark.parametrize(('args', 'upload_schema_str'), [
        ((arr, inp, ins, fmt) if all((inp, ins, fmt)) else
         (arr, inp, ins) if inp and ins else
         (arr, inp) if inp else
         (arr,), ups)
        for arr in ('foo', 'foo_not_null')
        for inp in (('', "'{fn}'") if arr else (None,))
        for ins in (('', '0') if inp else (None,))
        for fmt in (('', "'(int64)'", "'{fmt}'") if ins else (None,))
        for ups in (['<val:int64 not null>[i]'] +
                    ([None] if fmt == "'(int64)'" else []))
    ])
    def test_load_numpy_bytes(self, db, args, upload_schema_str):
        if args[0].endswith('not_null'):
            args = ('foo',) + args[1:]
            db.create_array('foo', '<val:int64 not null>[i]')
        else:
            db.create_array('foo', '<val:int64>[i]')
        assert type(
            db.load(
                *args,
                upload_data=foo_np.tobytes(),
                upload_schema=(Schema.fromstring(upload_schema_str)
                               if upload_schema_str else None))) == Array
        db.remove('foo')

    @pytest.mark.parametrize(('args', 'upload_schema_str'), [
        ((arr, inp, ins, fmt) if all((inp, ins, fmt)) else
         (arr, inp, ins) if inp and ins else
         (arr, inp) if inp else
         (arr,), ups)
        for arr in ('foo', 'foo_not_null')
        for inp in (('', "'{fn}'") if arr else (None,))
        for ins in (('', '0') if inp else (None,))
        for fmt in (('', "'(int64 null)'", "'{fmt}'") if ins else (None,))
        for ups in (['<val:int64>[i]'] +
                    ([None] if fmt == "'(int64 null)'" else []))
    ])
    def test_load_numpy_null_bytes(self, db, args, upload_schema_str):
        if args[0].endswith('not_null'):
            args = ('foo',) + args[1:]
            db.create_array('foo', '<val:int64 not null>[i]')
        else:
            db.create_array('foo', '<val:int64>[i]')
        assert type(
            db.load(
                *args,
                upload_data=foo_np_null.tobytes(),
                upload_schema=(Schema.fromstring(upload_schema_str)
                               if upload_schema_str else None))) == Array
        db.remove('foo')
