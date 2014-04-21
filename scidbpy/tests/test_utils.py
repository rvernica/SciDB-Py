# License: Simplified BSD, 2014
# See LICENSE.txt for more information

import numpy as np
from numpy.testing import assert_allclose
import itertools
from scidbpy.utils import broadcastable
from scidbpy._py3k_compat import genfromstr


def test_gen_from_string():
    s = '\n'.join(map(str, range(10)))
    a = genfromstr(s, dtype=float)
    assert_allclose(a, np.arange(10))


def test_broadcastable():
    for ndim1 in range(1, 4):
        for ndim2 in range(1, 4):
            for shape1 in itertools.permutations(range(1, 4), ndim1):
                for shape2 in itertools.permutations(range(1, 4), ndim2):
                    try:
                        np.broadcast(np.zeros(shape1),
                                     np.zeros(shape2))
                        result = True
                    except ValueError:
                        result = False
                    assert result == broadcastable(shape1, shape2)
