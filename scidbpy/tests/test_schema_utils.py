# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from __future__ import absolute_import, print_function, division, unicode_literals

from . import sdb, TestBase, teardown_function
from .. import schema_utils as su

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


class TestRedimension(TestBase):

    def test_dim_att_swap(self):
        x = sdb.arange(5) * 2
        y = su.redimension(x, ['x'], ['i0'])
        assert su.boundify(y, trim=True).size == 9

    def test_boolean_dim(self):
        x = sdb.from_array(np.array([True, False]))
        y = su.redimension(x, ['f0'], ['i0'])
        assert_array_equal(y.toarray(), [1, 0])


def test_new_alias_label():
    x = sdb.zeros(1)
    y = sdb.afl.cross_join(x.as_('L'), x.as_('R'),
                           'L.i0', 'R.i0')

    assert su.new_alias_label('B', x, y) == 'B'
    assert su.new_alias_label('L', x, y) != 'L'
    assert su.new_alias_label('R', x, y) != 'R'
