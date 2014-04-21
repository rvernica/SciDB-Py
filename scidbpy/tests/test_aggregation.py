# License: Simplified BSD, 2014
# See LICENSE.txt for more information

import pytest

from .. import connect, histogram
import numpy as np

sdb = connect()


class TestHistogram(object):

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
