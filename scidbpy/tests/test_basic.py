import numpy as np
from numpy.testing import assert_allclose

# In order to run tests, we need to connect to a valid SciDB engine
from scidbpy import interface
sdb = interface.SciDBShimInterface('http://localhost:8080')

def test_array_creation():
    def check_array_creation(create_array):
        # Create an array with 5x5 elements
        A = create_array((5, 5))
        name = A.name
        assert name in sdb.list_arrays()

        # when A goes out of scope, its data should be deleted from the engine
        del A
        assert name not in sdb.list_arrays()

    for create_array in [sdb.zeros, sdb.ones, sdb.random, sdb.randint]:
        yield check_array_creation, create_array


def test_numpy_conversion():
    x_in = np.random.random((10, 6, 5))
    x_sdb = sdb.from_array(x_in)
    x_out = x_sdb.toarray()
    assert_allclose(x_in, x_out)


def test_dot():
    A = sdb.random((4, 6))
    B = sdb.random((6, 5))
    C = sdb.dot(A, B)

    assert_allclose(C.toarray(), np.dot(A.toarray(), B.toarray()))

def test_svd():
    # chunk_size=32 currently required for svd
    A = sdb.random((6, 10), chunk_size=32)
    U, S, VT = sdb.svd(A)

    U2, S2, VT2 = np.linalg.svd(A.toarray(), full_matrices=False)

    assert_allclose(U.toarray(), U2)
    assert_allclose(S.toarray(), S2)
    assert_allclose(VT.toarray(), VT2)
