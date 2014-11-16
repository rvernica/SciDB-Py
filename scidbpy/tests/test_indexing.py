# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from __future__ import absolute_import, print_function, division, unicode_literals

from numpy.testing import assert_allclose, assert_array_equal
import numpy as np
import pytest

from .. import SciDBArray
from . import sdb, teardown_function, TestBase, RTOL
from .test_basic import needs_scipy


def test_slicing():
    # note that slices must be a divisor of chunk size

    def check_subarray(slc):
        A = sdb.random((10, 10), chunk_size=12)
        Aslc = A[slc]
        if isinstance(Aslc, SciDBArray):
            Aslc = Aslc.toarray()
        assert_allclose(Aslc, A.toarray()[slc], rtol=RTOL)

    for slc in [(slice(None), slice(None)),
                (2, 3),
                1,
                slice(2, 6),
                (slice(None), 2),
                (slice(2, 8), slice(3, 7)),
                (slice(2, 8, 2), slice(None, None, 3)),
                (slice(2, -2), slice(3, -2))]:
        yield check_subarray, slc

    # non-supported case
    #(slice(8, 2, -1), slice(7, 3, -1))


def test_slice_unbound_array():

    x = sdb.ones((3, 4)).redimension('<f0:double>[i0=0:*,10,0, i1=0:*,10,0]')
    y = x[0]
    assert_allclose(y.toarray(), [1, 1, 1, 1])
    assert list(y.datashape.dim_high) == [None]


class TestAttributeAccess(TestBase):

    def test_single(self):
        x = sdb.arange(5)
        assert_array_equal(x['f0'].toarray(), [0, 1, 2, 3, 4])

    def test_multi(self):
        x = sdb.arange(5)
        x = sdb.afl.apply(x, 'f1', 'f0+1')
        x = sdb.afl.apply(x, 'f2', 'f0+2')

        result = x[['f0', 'f1']].todataframe()
        assert_array_equal(result['f0'], [0, 1, 2, 3, 4])
        assert_array_equal(result['f1'], [1, 2, 3, 4, 5])

    def test_add_column(self):

        x = sdb.arange(5)
        x['f1'] = 'f0+1'
        assert_array_equal(x['f1'].toarray(), [1, 2, 3, 4, 5])

    def test_schema_updates_with_add_column(self):
        x = sdb.arange(5)
        x['f1'] = 'f0+1'
        assert x.att_names == ['f0', 'f1']

    def test_add_multi_columns(self):
        x = sdb.arange(5)
        x[['f1', 'f2']] = 'f0+1', 'f0+2'
        assert x.att_names == ['f0', 'f1', 'f2']

        assert_array_equal(x['f1'].toarray(), [1, 2, 3, 4, 5])
        assert_array_equal(x['f2'].toarray(), [2, 3, 4, 5, 6])


class TestBooleanIndexing(TestBase):

    def test_inequality_filter(self):

        def check(y):
            x = sdb.from_array(y)
            assert_array_equal((x[x < 5]).collapse().toarray(), y[y < 5])

        yield check, np.arange(12)
        yield check, np.arange(12).reshape(3, 4)
        yield check, np.arange(12).astype(np.float)

    def test_size_mismatch(self):

        with pytest.raises(ValueError) as exc:
            x = sdb.arange(5)
            y = sdb.from_array(np.array([True, False]))
            x[y]

        assert exc.value.args[0] == 'Shape of mask does not match array: (2,) vs (5,)'

    def test_numpy_boolean_mask(self):

        x = sdb.arange(10)
        y = np.arange(10) > 8

        assert_array_equal(x[y].collapse().toarray(), [9])

    def test_multiattribute(self):

        x = sdb.arange(5)
        x = sdb.afl.apply(x, 'y', 'f0+1')

        xnp = x.toarray()
        mnp = np.arange(5) > 3

        m = sdb.from_array(mnp)

        assert_array_equal(xnp[mnp], x[m].collapse().toarray())

    def test_name_collisions(self):
        x = sdb.arange(5)
        x = sdb.afl.apply(x, 'condition', 'f0+1', '__idx', 'f0+2')

        xnp = x.toarray()
        mnp = np.arange(5) > 3

        m = sdb.from_array(mnp)

        assert_array_equal(xnp[mnp], x[m].collapse().toarray())

    def test_index_preserved(self):
        x = sdb.arange(5)
        result = x[x > 3]
        att = result.dim_names[0]

        assert_array_equal(result.unpack('_').toarray()[att], [4])


class TestIndexIntegerArrays(TestBase):

    def test_1d(self):
        x = sdb.from_array(np.array([0, 10, 20]))
        y = sdb.from_array(np.array([0, 2, 0]))

        assert_array_equal(x[y].toarray(), x.toarray()[y.toarray()])

    def test_1d_multiattribute(self):
        x = sdb.random(5)
        y = sdb.random(5)
        z = sdb.join(x, y)
        idx = sdb.from_array(np.array([0, 0, 1, 1, 1]))

        assert_array_equal(z[idx].toarray(), z.toarray()[idx.toarray()])
