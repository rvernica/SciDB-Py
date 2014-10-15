from . import sdb, TestBase, teardown_function
from .. import schema_utils as su

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


class TestRedimension(TestBase):

    def test_dim_att_swap(self):
        x = sdb.arange(5) * 2
        y = su.redimension(x, ['x'], ['i0'])
        assert y.size == 9

    def test_boolean_dim(self):
        x = sdb.from_array(np.array([True, False]))
        y = su.redimension(x, ['f0'], ['i0'])
        assert_array_equal(y.toarray(), [1, 0])
