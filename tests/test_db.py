import numpy
import pytest

from scidbpy.db import connect, iquery


@pytest.fixture(scope='module')
def db():
    return connect()


class TestDB:

    @pytest.mark.parametrize('query', [
        'list()',
        "list('operators')",
    ])
    def test_iquery(self, db, query):
        assert db.iquery(query) == None
        assert type(db.iquery(query, fetch=True)) == numpy.ndarray

    @pytest.mark.parametrize('type_name', [
        '{}int{}'.format(pre, sz)
        for pre in ('', 'u')
        for sz in (8, 16, 32, 64)
    ] + [
        'bool',
        'float',
        'double',
    ])
    def test_fetch_numpy(self, db, type_name):
        ar = iquery(
            db,
            'build(<val:{}>[i=1:10,10,0], random())'.format(type_name),
            fetch=True)
        assert ar.shape == (10,)
        assert ar.ndim == 1

    @pytest.mark.parametrize('type_name', [
        '{}int{}'.format(pre, sz)
        for pre in ('', 'u')
        for sz in (8, 16, 32, 64)
    ] + [
        'bool',
        'float',
        'double',
    ])
    def test_fetch_dataframe(self, db, type_name):
        ar = iquery(
            db,
            'build(<val:{}>[i=1:10,10,0], random())'.format(type_name),
            fetch=True,
            as_dataframe=True)
        assert ar.shape == (10, 1)
        assert ar.ndim == 2
