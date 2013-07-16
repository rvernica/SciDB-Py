import numpy as np
import itertools
from scidbpy.utils import broadcastable


def test_broadcastable():
    for ndim1 in range(1, 4):
        for ndim2 in range(1, 4):
            for shape1 in itertools.permutations(range(1, 4), ndim1):
                for shape2 in itertools.permutations(range(1, 4), ndim2):
                    try:
                        b = np.broadcast(np.zeros(shape1),
                                         np.zeros(shape2))
                        result = True
                    except ValueError:
                        result = False
                    assert result == broadcastable(shape1, shape2)
