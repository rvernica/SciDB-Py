import numpy
import pytest

from scidbpy.db import connect


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
