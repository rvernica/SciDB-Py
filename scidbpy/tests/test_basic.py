import numpy as np
from numpy.testing import assert_allclose

from nose import SkipTest

# In order to run tests, we need to connect to a valid SciDB engine
from scidbpy import interface, SciDBQueryError
sdb = interface.SciDBShimInterface('http://localhost:8080')

RTOL = 1E-6


def test_numpy_conversion():
    X = np.random.random((10, 6))

    Xsdb = sdb.from_array(X)

    def check_toarray(transfer_bytes):
        Xnp = Xsdb.toarray(transfer_bytes=transfer_bytes)
        # set ATOL high because we're translating text
        assert_allclose(Xnp, X, atol=1E-5)

    for transfer_bytes in (True, False):
        yield check_toarray, transfer_bytes


def test_toarray_sparse():
    try:
        from scipy import sparse
    except:
        raise SkipTest("scipy.sparse required for this test")

    X = np.random.random((10, 6))
    Xsdb = sdb.from_array(X)
    Xcsr = Xsdb.tosparse('csr')
    assert_allclose(X, Xcsr.toarray())


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


def test_raw_query():
    """Test a more involved raw query: creating a tri-diagonal matrix"""
    arr = sdb.new_array((10, 10))
    sdb.query('store(build({A},iif({A.d0}={A.d1},2,'
              'iif(abs({A.d0}-{A.d1})=1,1,0))),{A})',
              A=arr)

    # Build the numpy equivalent
    np_arr = np.zeros((10, 10))
    np_arr.flat[0::11] = 2  # set diagonal to 2
    np_arr.flat[1::11] = 1  # set upper off-diagonal to 1
    np_arr.flat[10::11] = 1  # set lower off-diagonal to 1

    assert_allclose(arr.toarray(), np_arr, rtol=RTOL)


def test_identity():
    def check_identity(n, sparse):
        I = sdb.identity(n, sparse=sparse)
        assert_allclose(I.toarray(), np.identity(n))

    for sparse in (True, False):
        yield check_identity, 6, sparse


def test_dot():
    A = sdb.random((4, 6))
    B = sdb.random((6, 5))
    C = sdb.dot(A, B)

    assert_allclose(C.toarray(), np.dot(A.toarray(), B.toarray()), rtol=RTOL)


def test_svd():
    # chunk_size=32 currently required for svd
    A = sdb.random((6, 10), chunk_size=32)

    try:
        U, S, VT = sdb.svd(A)
    except SciDBQueryError:
        # SVD is not part of the default install... skip the test
        raise SkipTest("SVD is not supported on your system")

    U2, S2, VT2 = np.linalg.svd(A.toarray(), full_matrices=False)

    assert_allclose(U.toarray(), U2, rtol=RTOL)
    assert_allclose(S.toarray(), S2, rtol=RTOL)
    assert_allclose(VT.toarray(), VT2, rtol=RTOL)


def test_subarray():
    # note that slices must be a divisor of chunk size
    A = sdb.random((10, 10), chunk_size=12)
    def check_subarray(slc1, slc2):
        Aslc = A[slc1, slc2]
        assert_allclose(Aslc.toarray(), A.toarray()[slc1, slc2], rtol=RTOL)

    for (slc1, slc2) in [(slice(None), slice(None)),
                         (slice(2, 8), slice(3, 7)),
                         (slice(2, 8, 2), slice(None, None, 3))]:
        yield check_subarray, slc1, slc2


def test_ops():
    from operator import add, sub, mul, div, mod, pow
    A = sdb.random((5, 5))
    B = 1.2

    def check_join_op(op):
        C = op(A, B)
        assert_allclose(C.toarray(), op(A.toarray(), B), rtol=RTOL)

    for op in (add, sub, mul, div, mod):
        yield check_join_op, op


def test_join_ops():
    from operator import add, sub, mul, div, mod
    A = sdb.random((5, 5))
    B = sdb.random((5, 5))

    def check_join_op(op):
        C = op(A, B)
        assert_allclose(C.toarray(), op(A.toarray(), B.toarray()), rtol=RTOL)

    for op in (add, sub, mul, div, mod):
        yield check_join_op, op


def test_abs():
    A = sdb.random((5, 5))
    B = abs(A - 1)
    assert_allclose(B.toarray(), abs(A.toarray() - 1))


def test_transcendentals():
    def np_op(op):
        D = dict(asin='arcsin', acos='arccos', atan='arctan')
        return D.get(op, op)

    A = sdb.random((5, 5))

    def check_op(op):
        C = getattr(sdb, op)(A)
        C_np = getattr(np, np_op(op))(A.toarray())
        assert_allclose(C.toarray(), C_np, rtol=RTOL)

    for op in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan',
               'exp', 'log', 'log10']:
        yield check_op, op


def test_aggregates():
    A = sdb.random((5, 5))

    ind_dict = {1:0, 0:1, None:None}

    def check_op(op, ind):
        C = getattr(sdb, op)(A, ind)
        if op in ['var', 'std']:
            C_np = getattr(np, op)(A.toarray(), ind_dict[ind], ddof=1)
        else:
            C_np = getattr(np, op)(A.toarray(), ind_dict[ind])
        assert_allclose(C.toarray(), C_np, rtol=1E-6)

    for op in ['min', 'max', 'sum', 'var', 'std', 'mean']:
        for ind in [None, 0, 1]:
            yield check_op, op, ind


#def test_pairwise_distances():
#    A = sdb.random((5, 3))
#    B = sdb.random((4, 3))
#
#    D = sdb.pairwise_distances(A, B)
#    D_np = np.sqrt(np.sum((A.toarray()[:, None, :] - B.toarray()) ** 2, -1))
#
#    assert_allclose(D.toarray(), D_np, rtol=RTOL)


def test_transpose():
    A = sdb.random((5, 3))
    assert_allclose(A.toarray().T, A.T.toarray(), rtol=RTOL)

def test_join():
    A = sdb.randint(10)
    B = sdb.randint(10)
    C = sdb.join(A, B)

    Cnp = C.toarray()
    names = Cnp.dtype.names
    assert_allclose(Cnp[names[0]], A.toarray())
    assert_allclose(Cnp[names[1]], B.toarray())
