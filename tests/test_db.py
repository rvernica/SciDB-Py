import numpy
import pytest

from scidbpy.db import connect, iquery


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
        assert db.iquery(query) == None
        assert type(db.iquery(query, fetch=True)) == numpy.ndarray

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
            schema=schema)
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
    @pytest.mark.parametrize('index', [
        False,
        None,
        [],
    ])
    def test_fetch_dataframe(self, db, type_name, index, schema):
        # Pandas DataFrame
        ar = iquery(
            db,
            'build(<val:{}>[i=1:10,10,0], random())'.format(type_name),
            fetch=True,
            as_dataframe=True,
            index=index,
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
    @pytest.mark.parametrize('index', [
        False,
        None,
        [],
    ])
    def test_fetch_dataframe_atts(self, db, type_name, index, schema):
        # Pandas DataFrame
        ar = iquery(
            db,
            'build(<val:{}>[i=1:10,10,0], random())'.format(type_name),
            fetch=True,
            atts_only=True,
            as_dataframe=True,
            index=index,
            schema=schema)
        assert ar.shape == (10, 1)
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
    @pytest.mark.parametrize('index', [
        True,
        ['val'],
    ])
    def test_fetch_dataframe_index(self, db, type_name, index, schema):
        # Pandas DataFrame, index
        ar = iquery(
            db,
            'build(<val:{}>[i=1:10,10,0], random())'.format(type_name),
            fetch=True,
            as_dataframe=True,
            index=index,
            schema=schema)
        assert ar.shape == (10, 1)
        assert ar.ndim == 2
