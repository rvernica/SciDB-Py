# License: Simplified BSD, 2014
# See LICENSE.txt for more information

import pytest
from numpy.testing import assert_allclose
import numpy as np

from .. import histogram
from . import sdb, TestBase, teardown_function


class TestHistogram(TestBase):

    def setup_method(self, method):
        np.random.seed(42)

    def test_bad_input(self):
        with pytest.raises(TypeError):
            histogram(1)

    def check(self, x, **kwargs):
        s = sdb.from_array(x)
        counts, bins = histogram(s, **kwargs)
        excounts, exbins = np.histogram(x, **kwargs)

        np.testing.assert_array_almost_equal(bins, exbins)
        np.testing.assert_array_equal(counts, excounts)

    def check_multi(self, x, att, **kwargs):
        s = sdb.from_array(x)
        counts, bins = histogram(s, att=att, **kwargs)
        excounts, exbins = np.histogram(x[att], **kwargs)
        np.testing.assert_array_almost_equal(bins, exbins)
        np.testing.assert_array_equal(counts, excounts)

    def test_defaults(self):
        x = np.random.random(100)
        self.check(x)

    def test_nbins(self):
        x = np.random.random(50)
        self.check(x, bins=17)

    def test_range(self):
        x = np.random.random(50)
        self.check(x, range=[0, 3])

    def test_multidimensional(self):
        x = np.random.random((5, 8))
        self.check(x)

    def test_multiatribute(self):
        x = np.zeros((3, 4),
                     dtype=[('x', '<f8'), ('y', '<f8')])
        x['x'] = np.random.random((3, 4))
        x['y'] = np.random.random((3, 4))
        self.check_multi(x, 'x')

    def test_nameconflict(self):
        x = np.zeros((3, 4),
                     dtype=[('bin', '<f8'), ('counts', '<f8')])
        x['bin'] = np.random.random((3, 4))
        x['counts'] = np.random.random((3, 4))
        self.check_multi(x, 'bin')
        self.check_multi(x, 'counts')

    def test_integer(self):
        x = np.random.randint(0, 5, 10)
        self.check(x)

    def test_integer_multiarray(self):
        x = np.zeros((3, 4),
                     dtype=[('a', np.int), ('b', np.float)])
        x['a'] = np.random.randint(0, 5, (3, 4))
        x['b'] = np.random.random((3, 4))
        self.check_multi(x, 'a')
        self.check_multi(x, 'b')


class TestGroupBy(object):

    def setup_method(self, method):
        self.a = sdb.afl.build('<val:int32>[i=0:5,10,0, j=0:3,10,0]', 2)
        self.b = sdb.afl.build('<k:float>[i=0:5,10,0, j=0:3,10,0]', 1)
        self.c = sdb.join(self.a, self.b)

    def teardown_method(self, method):
        sdb.reap()

    def test_global_groupby(self):

        x = self.a.groupby('i').aggregate('sum(val)').toarray()
        assert_allclose(x['val_sum'], [8, 8, 8, 8, 8, 8])

        x = self.a.groupby('j').aggregate('count(*)').toarray()
        assert_allclose(x['count'], [6, 6, 6, 6])

    def test_groupby_on_multi_dims(self):

        x = self.a.groupby(['i', 'j']).aggregate('count(*)').toarray()
        assert_allclose(x['count'], np.ones(24))

    def test_groupby_mapping(self):

        x = self.a.groupby(['i', 'j']).aggregate({'c': 'count(*)'}).toarray()
        assert_allclose(x['c'], np.ones(24))

    def test_multimapping(self):
        x = self.a.groupby('j').aggregate({'c': 'count(*)', 's': 'sum(val)'}).toarray()
        assert_allclose(x['c'], np.ones(4) * 6)
        assert_allclose(x['s'], np.ones(4) * 12)

    def test_group_on_attriute(self):
        x = self.c.groupby('val').aggregate('sum(k)').toarray()
        assert_allclose(x['val'], [2])
        assert_allclose(x['k_sum'], [24])

    def test_float_forbidden(self):
        with pytest.raises(TypeError) as exc:
            self.b.groupby('k')
        assert exc.value.args[0] == 'Grouping by non-integer attributes not yet supported'

    def test_group_over_all_attributes(self):
        x = self.a.groupby('val').aggregate('count(*)').toarray()
        assert_allclose(x['count'], [24])

    # other tests: handle aggregation over a dimension?
