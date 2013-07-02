import numpy as np
from scidbpy.scidbarray import SDB_TYPE_LIST, SDB_NP_TYPE_MAP, sdbtype


def test_sdbtype_dtype_mapping():
    """Test the mapping of SciDB types to numpy types"""
    for i in range(len(SDB_TYPE_LIST) - 3):
        dtype = [('val{0}'.format(j), SDB_NP_TYPE_MAP[SDB_TYPE_LIST[j]])
                 for j in range(i, min(len(SDB_TYPE_LIST), i + 3))]
        dtype_start = np.dtype(dtype)
        assert(dtype_start == sdbtype(dtype_start).dtype)


if __name__ == '__main__':
    import nose
    nose.runmodule()
