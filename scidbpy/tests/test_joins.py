from __future__ import print_function, absolute_import, unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from . import sdb, TestBase, teardown_function, randarray, needs_pandas
from scidbpy.relational import merge

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


@needs_pandas
class TestMerge(TestBase):

    def check(self, expected, actual):
        actual = actual.todataframe()

        print(actual)
        print(expected)

        # don't care about column order
        assert set(expected.columns) == set(actual.columns)
        expected = expected[actual.columns]

        # rowsort by index, we don't care about order
        actual = actual.sort(actual.columns.values.tolist())
        expected = expected.sort(expected.columns.values.tolist())

        # don't care about index
        expected.index = np.arange(len(expected))
        expected.index.name = actual.index.name

        print(actual)
        print(expected)

        assert_frame_equal(expected, actual)

    def test_dimension_merge(self):

        a = pd.DataFrame(dict(x=[1, 2, 3], y=[10, 20, 30]))
        b = pd.DataFrame(dict(z=[20, 30, 40]))

        expected = pd.DataFrame(dict(x=[1, 2, 3],
                                y=[10, 20, 30],
                                z=[20, 30, 40],
                                index=[0, 1, 2]))
        actual = merge(sdb.from_dataframe(a), sdb.from_dataframe(b),
                       on='index')

        self.check(expected, actual)

    def test_default_merge(self):
        a = pd.DataFrame(dict(x=[1, 2, 3], y=[10, 20, 30]))
        b = pd.DataFrame(dict(x=[1, 5, 3], z=[20, 30, 40]))

        expected = pd.DataFrame(dict(x=[1, 3],
                                y=[10, 30],
                                z=[20, 40],
                                index=[0, 2]))
        actual = merge(sdb.from_dataframe(a), sdb.from_dataframe(b))

        self.check(expected, actual)

    def test_multicolumn_merge(self):
        a = pd.DataFrame(dict(x=[1, 2, 3], y=[10, 20, 30], w=[4, 5, 6]))
        b = pd.DataFrame(dict(x=[1, 5, 3], z=[20, 30, 40], w=[4, 6, 5]))

        expected = pd.DataFrame(dict(x=[1], y=[10], z=[20], w=[4],
                                index=[0]))
        actual = merge(sdb.from_dataframe(a), sdb.from_dataframe(b))

        self.check(expected, actual)

    def test_attribute_merge(self):

        a = pd.DataFrame(dict(x=[1, 2, 3], y=[10, 20, 30]))
        b = pd.DataFrame(dict(x=[1, 5, 3], z=[20, 30, 40]))

        expected = pd.DataFrame(dict(x=[1, 3],
                                     y=[10, 30],
                                     z=[20, 40],
                                     index_y=[0, 2],
                                     index_x=[0,2]))
        actual = merge(sdb.from_dataframe(a), sdb.from_dataframe(b), on='x')

        self.check(expected, actual)


    def test_manyone(self):

        a = pd.DataFrame(dict(x=[1, 2, 3, 1], y=[10, 20, 30, 40]))
        b = pd.DataFrame(dict(x=[1, 5, 3], z=[20, 30, 40]))

        expected = pd.DataFrame(dict(x=[1, 1, 3],
                                     y=[10, 40, 30],
                                     z=[20, 20, 40],
                                     index_y=[0, 0, 2],
                                     index_x=[0, 3, 2]))
        actual = merge(sdb.from_dataframe(a), sdb.from_dataframe(b), on='x')

        self.check(expected, actual)

    def test_manymany(self):

        a = pd.DataFrame(dict(x=[1, 2, 3, 1], y=[10, 20, 30, 40]))
        b = pd.DataFrame(dict(x=[1, 1, 5, 3], z=[0, 20, 30, 40]))

        expected = pd.DataFrame(dict(x=[1, 1, 1, 1, 3],
                                     y=[10, 40, 10, 40, 30],
                                     z=[0, 0, 20, 20, 40],
                                     index_y=[0, 0, 1, 1, 3],
                                     index_x=[0, 3, 0, 3, 2]))
        actual = merge(sdb.from_dataframe(a), sdb.from_dataframe(b),
                       on='x')
        self.check(expected, actual)

    @pytest.mark.skipif(True, reason="Not implemented yet")
    def test_left(self):

        a = pd.DataFrame(dict(x=[1, 2, 3], y=[10, 20, 30]))
        b = pd.DataFrame(dict(x=[1, 5, 3], z=[20, 30, 40]))

        expected = pd.DataFrame(dict(x=[1, 2, 3],
                                     y=[10, 20, 30],
                                     z=[20, np.nan, 40],
                                     index_y=[0, np.nan, 2],
                                     index_x=[0, 1, 2]))
        actual = merge(sdb.from_dataframe(a), sdb.from_dataframe(b),
                       how='left', on='x')

        self.check(expected, actual)

    def test_boolean_join(self):

        a = pd.DataFrame(dict(x=[True, False], y=[10, 20]))
        b = pd.DataFrame(dict(x=[False, True], z=[20, 30]))

        expected = pd.DataFrame(dict(x=[True, False],
                                     y=[10, 20],
                                     z=[30, 20],
                                     index_1=[0, 1],
                                     index_2=[1, 0]))
        actual = merge(sdb.from_dataframe(a), sdb.from_dataframe(b),
                       on='x', suffixes=('_1', '_2'))

        self.check(expected, actual)

    def test_string_join(self):
        a = np.array([("one", 10), ("two", 20)],
                     dtype=[(str('x'), '|S3'), (str('y'), int)])
        b = np.array([("two", 30), ("one", 40)],
                     dtype=[(str('x'), '|S3'), (str('z'), int)])

        expected = pd.DataFrame(dict(x=["one", "two"],
                                     y=[10, 20],
                                     z=[40, 30],
                                     i0_1=[0, 1],
                                     i0_2=[1, 0]))
        actual = merge(sdb.from_array(a), sdb.from_array(b),
                       on='x', suffixes=('_1', '_2'))

        self.check(expected, actual)

    def test_string_join_extra_cells(self):
        a = np.array([("one", 10), ("two", 20), ("three", 30)],
                     dtype=[(str('x'), '|S8'), (str('y'), int)])
        b = np.array([("two", 30), ("five", 50), ("one", 40)],
                     dtype=[(str('x'), '|S8'), (str('z'), int)])

        expected = pd.DataFrame(dict(x=["one", "two"],
                                     y=[10, 20],
                                     z=[40, 30],
                                     i0_1=[0, 1],
                                     i0_2=[2, 0]))
        actual = merge(sdb.from_array(a), sdb.from_array(b),
                       on='x', suffixes=('_1', '_2'))

        self.check(expected, actual)

    def test_separate_on(self):

        a = pd.DataFrame(dict(x=[True, False], y=[10, 20]))
        b = pd.DataFrame(dict(a=[False, True], z=[20, 30]))

        expected = pd.DataFrame(dict(x=[True, False],
                                     y=[10, 20],
                                     z=[30, 20],
                                     index_a=[0, 1],
                                     index_b=[1, 0]))
        actual = merge(sdb.from_dataframe(a), sdb.from_dataframe(b),
                       left_on='x',
                       right_on='a', suffixes=('_a', '_b'))
        self.check(expected, actual)

    def test_books(self):

        authors = np.array([('Tukey', 'US', True),
                           ('Venables', 'Australia', False),
                           ('Tierney', 'US', False),
                           ('Ripley', 'UK', False),
                           ('McNeil', 'Australia', False)],
                           dtype=[(str('surname'), 'S10'), (str('nationality'), 'S10'),
                                  (str('deceased'), '?')])
        books = np.array([('Exploratory Data Analysis', 'Tukey'),
                         ('Modern Applied Statistics ...', 'Venables'),
                         ('LISP-STAT', 'Tierney'),
                         ('Spatial Statistics', 'Ripley'),
                         ('Stochastic Simulation', 'Ripley'),
                         ('Interactive Data Analysis', 'McNeil'),
                         ('Python for Data Analysis', 'McKinney')],
                         dtype=[(str('title'), 'S40'), (str('name'), 'S10')])
        expected = pd.DataFrame(dict(i0_x=[0, 1, 2, 3, 3, 4],
                                     i0_y=[0, 1, 2, 3, 4, 5],
                                     surname=['Tukey', 'Venables', 'Tierney', 'Ripley', 'Ripley', 'McNeil'],
                                     title=books['title'][:-1],
                                     nationality=['US', 'Australia', 'US', 'UK', 'UK', 'Australia'],
                                     deceased=[True, False, False, False, False, False]))

        actual = merge(sdb.from_array(authors), sdb.from_array(books),
                       left_on='surname', right_on='name')
        self.check(expected, actual)
