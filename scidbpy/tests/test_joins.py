# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from __future__ import absolute_import, print_function, division, unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_allclose

from . import sdb, TestBase, teardown_function, needs_pandas
from scidbpy.relational import merge
from .. import schema_utils as su

try:
    import pandas as pd
    from pandas.util.testing import assert_frame_equal
except ImportError:
    pass


def test_join():
    A = sdb.randint(10)
    B = sdb.randint(10)
    C = sdb.join(A, B)

    Cnp = C.toarray()
    names = Cnp.dtype.names
    assert_allclose(Cnp[names[0]], A.toarray())
    assert_allclose(Cnp[names[1]], B.toarray())


def test_cross_join():
    A = sdb.random((10, 5))
    B = sdb.random(10)
    AB = sdb.cross_join(A, B, (0, 0))

    ABnp = AB.toarray()
    names = ABnp.dtype.names
    assert_allclose(ABnp[names[0]],
                    A.toarray())
    assert_allclose(ABnp[names[1]],
                    B.toarray()[:, None] + np.zeros(A.shape[1]))


def make(dims, **arrays):
    n = len(list(arrays.values())[0])
    row = list(range(n))
    arrays.setdefault('row', row)

    df = pd.DataFrame(arrays)
    result = sdb.from_dataframe(df)

    df = df.set_index(list(dims))
    result = su.boundify(su.redimension(result, dims, list(df.columns.values)), trim=True)
    return result.eval()


@needs_pandas
class TestMerge(TestBase):

    def check(self, expected, actual):
        actual = actual.todataframe()

        print("Actual")
        print(actual)
        print("Expected")
        print(expected)

        # don't care about column order
        assert set(expected.columns) == set(actual.columns)
        expected = expected[actual.columns]

        # rowsort by index, we don't care about order
        actual = actual.sort(actual.columns.values.tolist())
        expected = expected.sort(expected.columns.values.tolist())

        assert_frame_equal(expected, actual)

    def test_dimension_merge(self):

        a = make(('row',), x=[1, 2, 3], y=[10, 20, 30])
        b = make(('row',), z=[20, 30, 40])

        expected = pd.DataFrame(dict(x=[1, 2, 3],
                                     y=[10, 20, 30],
                                     z=[20, 30, 40]))
        expected.index.name = 'row'
        actual = merge(a, b, on='row')

        self.check(expected, actual)

    def test_default_merge(self):
        a = make(('row', 'a'), x=[1, 2, 3], y=[10, 20, 30], a=[0, 0, 0])
        b = make(('row', 'b'), x=[1, 5, 3], z=[20, 30, 40], b=[0, 0, 0])

        expected = pd.DataFrame(dict(x_x=[1, 2, 3],
                                     x_y=[1, 5, 3],
                                     a=[0, 0, 0],
                                     b=[0, 0, 0],
                                     y=[10, 20, 30],
                                     row=[0, 1, 2],
                                     z=[20, 30, 40])).set_index(['row', 'a', 'b'])
        actual = merge(a, b)

        self.check(expected, actual)

    def _test_multicolumn_merge(self):
        a = pd.DataFrame(dict(x=[1, 2, 3], y=[10, 20, 30], w=[4, 5, 6]))
        b = pd.DataFrame(dict(x=[1, 5, 3], z=[20, 30, 40], w=[4, 6, 5]))

        expected = pd.DataFrame(dict(x=[1], y=[10], z=[20], w=[4],
                                     index=[0]))
        actual = merge(a, b)

        self.check(expected, actual)

    def test_attribute_merge(self):

        a = make(('row',), x=[1, 2, 3], y=[10, 20, 30])
        b = make(('row',), x=[1, 5, 3], z=[20, 30, 40])

        expected = pd.DataFrame(dict(x_x=[1, 3],
                                     x_y=[1, 3],
                                     y=[10, 30],
                                     z=[20, 40],
                                     x_cat=[0, 2],
                                     row_y=[0, 2],
                                     row_x=[0, 2])).set_index(['row_x', 'x_cat', 'row_y'])
        expected.index.name = 'row'
        actual = merge(a, b, on='x')

        self.check(expected, actual)

    def test_manyone(self):
        a = make(('x', 'row'), x=[1, 2, 3, 1], y=[10, 20, 30, 40])
        b = make(('x', 'row'), x=[1, 0, 3, 0], z=[20, 30, 40, 80])
        expected = pd.DataFrame(dict(x=[1, 1, 3],
                                     y=[10, 40, 30],
                                     z=[20, 20, 40],
                                     row_x=[0, 3, 2],
                                     row_y=[0, 0, 2])).set_index(['x', 'row_x', 'row_y'])

        actual = merge(a, b, on='x')
        self.check(expected, actual)

    def test_manymany(self):

        a = make(('x', 'row'), x=[1, 2, 3, 1], y=[10, 20, 30, 40])
        b = make(('x', 'row'), x=[1, 1, 5, 3], z=[0, 20, 30, 40])

        expected = pd.DataFrame(dict(x=[1, 1, 1, 1, 3],
                                     y=[10, 40, 10, 40, 30],
                                     z=[0, 0, 20, 20, 40],
                                     row_y=[0, 0, 1, 1, 3],
                                     row_x=[0, 3, 0, 3, 2])).set_index(['x', 'row_x', 'row_y'])
        actual = merge(a, b,
                       on='x')
        self.check(expected, actual)

    def test_string_join(self):
        a = np.array([("one", 10), ("two", 20)],
                     dtype=[(str('x'), '|S3'), (str('y'), int)])
        b = np.array([("two", 30), ("one", 40)],
                     dtype=[(str('x'), '|S3'), (str('z'), int)])

        expected = pd.DataFrame(dict(x_1=["one", "two"],
                                     x_2=["one", "two"],
                                     y=[10, 20],
                                     x_cat=[0, 1],
                                     z=[40, 30],
                                     i0_1=[0, 1],
                                     i0_2=[1, 0])).set_index(['i0_1', 'x_cat', 'i0_2'])
        actual = merge(sdb.from_array(a), sdb.from_array(b),
                       on='x', suffixes=('_1', '_2'))

        self.check(expected, actual)

    def test_string_join_extra_cells(self):
        a = np.array([("one", 10), ("two", 20), ("three", 30)],
                     dtype=[(str('x'), '|S8'), (str('y'), int)])
        b = np.array([("two", 30), ("five", 50), ("one", 40)],
                     dtype=[(str('x'), '|S8'), (str('z'), int)])

        expected = pd.DataFrame(dict(x_1=["one", "two"],
                                     x_2=["one", "two"],
                                     x_cat=[0, 2],
                                     y=[10, 20],
                                     z=[40, 30],
                                     i0_1=[0, 1],
                                     i0_2=[2, 0])).set_index(['i0_1', 'x_cat', 'i0_2'])
        actual = merge(sdb.from_array(a), sdb.from_array(b),
                       on='x', suffixes=('_1', '_2'))

        self.check(expected, actual)

    def test_separate_on(self):
        a = make(('i', 'row'), i=[1, 2], l=[2, 3])
        b = make(('j', 'row'), j=[1, 2], k=[2, 3])

        expected = pd.DataFrame(dict(i=[1, 2],
                                     row_a=[0, 1],
                                     row_b=[0, 1],
                                     l=[2, 3],
                                     k=[2, 3])).set_index(['i', 'row_a', 'row_b'])
        actual = merge(a, b,
                       left_on='i',
                       right_on='j', suffixes=('_a', '_b'))
        self.check(expected, actual)

    def test_dimension_attribute_conflict(self):

        a = make(('i', 'row'), i=[1, 2], j=[2, 3])
        b = make(('j', 'row'), j=[1, 2], k=[2, 3])

        expected = pd.DataFrame(dict(i=[1, 2],
                                     j_a=[2, 3],
                                     row_a=[0, 1],
                                     row_b=[0, 1],
                                     k=[2, 3])).set_index(['i', 'row_a', 'row_b'])
        actual = merge(a, b,
                       left_on='i',
                       right_on='j', suffixes=('_a', '_b'))
        self.check(expected, actual)


@needs_pandas
class TestBadMerges(TestBase):

    def test_not_implemented_ifnot_inner(self):
        x = sdb.ones(1)

        with pytest.raises(NotImplementedError):
            sdb.merge(x, x, how='left')

        with pytest.raises(NotImplementedError):
            sdb.merge(x, x, how='right')

        with pytest.raises(NotImplementedError):
            sdb.merge(x, x, how='outer')

    def test_too_many_ons(self):

        x = sdb.ones(1)

        with pytest.raises(ValueError) as exc:
            sdb.merge(x, x, left_on='f0', right_on='f0', on='f0')
        assert exc.value.args[0] == 'Cannot specify left_on/right_on with on'

    def test_mismatched_left_right(self):

        x = sdb.ones(1)
        with pytest.raises(ValueError) as exc:
            sdb.merge(x, x, left_on=['a', 'b'], right_on=['c'])
        assert exc.value.args[0] == 'left_on and right_on must have the same number of items'

    def test_invalid_left(self):
        x = sdb.ones(1)

        with pytest.raises(ValueError) as exc:
            sdb.merge(x, x, left_on='nope', right_on=x.dim_names[0])
        assert exc.value.args[0] == 'Left array join name is invalid: nope'

    def test_invalid_right(self):
        x = sdb.ones(1)

        with pytest.raises(ValueError) as exc:
            sdb.merge(x, x, right_on='nope', left_on=x.dim_names[0])
        assert exc.value.args[0] == 'Right array join name is invalid: nope'

    def test_dim_attribute_join(self):

        x = sdb.ones(1)

        with pytest.raises(NotImplementedError) as exc:
            sdb.merge(x, x, left_on=x.dim_names, right_on=x.att_names)
