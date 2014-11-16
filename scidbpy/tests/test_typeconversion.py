# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np
from scidbpy.scidbarray import SDB_NP_TYPE_MAP, sdbtype


def test_sdbtype_dtype_mapping():
    """Test the mapping of SciDB types to numpy types"""
    type_list = list(SDB_NP_TYPE_MAP.keys())
    for i in range(len(type_list) - 3):
        dtype = [(str('val{0}'.format(j)), SDB_NP_TYPE_MAP[type_list[j]])
                 for j in range(i, min(len(type_list), i + 3))]
        dtype_start = np.dtype(dtype)
        assert(dtype_start == sdbtype(dtype_start).dtype)
