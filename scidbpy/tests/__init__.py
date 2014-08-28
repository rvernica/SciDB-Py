# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from numpy.random import randint


from .. import connect
from ..robust import rechunk

sdb = connect()

# NOTE: this needs to be explicitly imported in each test module,
#      or pytest won't run it


def teardown_function(function):
    sdb.reap()


class TestBase(object):

    def teardown_method(self, method):
        sdb.reap()


"""
The code below randomly rechunks arrays created
by standard factory methods like sdb.zeros()

This tests that SciDBPy functions are robust to details about chunks
"""


def chunk_fuzz(func):

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        persistent = result.persistent
        result.persistent = False
        chunks = randint(100, 300, (result.ndim,))
        chunk_overlap = randint(0, 3, (result.ndim,))
        result = rechunk(result, chunk_size=chunks, chunk_overlap=chunk_overlap)
        result.eval()
        result.persistent = persistent
        return result

    return wrapper

for fac in ('zeros ones random from_array from_sparse from_dataframe '
            'identity linspace arange random randint').split():
    setattr(sdb, fac, chunk_fuzz(getattr(sdb, fac)))
