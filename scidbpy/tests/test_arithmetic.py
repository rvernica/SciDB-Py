# -*- coding: utf-8 -*-

# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from operator import add, sub, mul, truediv, mod, pow

from numpy.testing import assert_allclose

from . import sdb, teardown_function, TestBase
from .test_basic import needs_scipy

RTOL = 1e-6
ARITHMETIC_OPERATORS = (add, sub, mul, truediv, mod, pow)


def test_ops():
    """
    Array [op] scalar
    """
    def check_join_op(op):
        A = sdb.random((5, 5))
        B = 1.2
        C = op(A, B)
        assert_allclose(C.toarray(), op(A.toarray(), B), rtol=RTOL)

    for op in ARITHMETIC_OPERATORS:
        yield check_join_op, op


def test_reverse_ops():
    """
    scalar [op] array
    """

    def check_join_op(op):
        A = 1.2
        B = sdb.random((5, 5))
        C = op(A, B)
        assert_allclose(C.toarray(), op(A, B.toarray()), rtol=RTOL)

    for op in ARITHMETIC_OPERATORS:
        yield check_join_op, op


def test_join_ops():
    """
    array [op] array2

    Matching shape, dense
    """

    def check_join_op(op):
        A = sdb.random((5, 5))
        B = sdb.random((5, 5))
        C = op(A, B)
        assert_allclose(C.toarray(), op(A.toarray(), B.toarray()), rtol=RTOL)

    for op in ARITHMETIC_OPERATORS:
        yield check_join_op, op


@needs_scipy
def test_sparse_joins():
    """
    array [op] other

    sparse matrices with different sparsity patterns
    """
    from scipy.sparse import rand

    def check_join_op(op):
        A = rand(3, 4, density=0.5)
        B = rand(3, 4, density=0.5)
        C = op(sdb.from_sparse(A), sdb.from_sparse(B))
        expected = op(A.toarray(), B.toarray())
        assert_allclose(C.toarray(), expected, rtol=RTOL)

    # ignore pow: needs square arrays
    for op in ARITHMETIC_OPERATORS[:-1]:
        yield check_join_op, op


@needs_scipy
def test_sparse_scalar():
    """
    sparse array [op] scalar
    scalar [op] sparse array
    """
    from scipy.sparse import rand

    def check_join_op(op):
        A = rand(3, 4, density=0.5)
        B = 1.2
        C = op(sdb.from_sparse(A), B)
        print C.query
        expected = op(A.toarray(), B)
        assert_allclose(C.toarray(), expected, rtol=RTOL)

        C = op(B, sdb.from_sparse(A))
        print C.query
        expected = op(B, A.toarray())
        assert_allclose(C.toarray(), expected, rtol=RTOL)

    # ignore pow: needs square arrays
    for op in ARITHMETIC_OPERATORS[:-1]:
        yield check_join_op, op


@needs_scipy
def test_sparse_mischunked():
    """
    array [op] other

    sparse matrices with different sparsity patterns and different
    dim schemas
    """
    from scipy.sparse import rand

    def check_join_op(op):
        A = rand(3, 4, density=0.5)
        B = rand(3, 4, density=0.5)
        Asdb = sdb.from_sparse(A).redimension('<f0:double>[i0=0:2,10,0,i1=0:3,10,0]')
        Bsdb = sdb.from_sparse(B).redimension('<f0:double>[i0=0:2,2,1,i1=0:3,2,1]')
        C = op(Asdb, Bsdb)
        expected = op(A.toarray(), B.toarray())
        assert_allclose(C.toarray(), expected, rtol=RTOL)

    # ignore pow: needs square arrays
    for op in ARITHMETIC_OPERATORS[:-1]:
        yield check_join_op, op


def test_dense_mischunked():

    def check_join_op(op):
        A = sdb.random((3, 4)).redimension('<f0:double>[i0=0:2,10,0,i1=0:3,10,0]')
        B = sdb.random((3, 4)).redimension('<f0:double>[i0=0:2,2,1,i1=0:3,2,1]')
        C = op(A, B)
        expected = op(A.toarray(), B.toarray())
        assert_allclose(C.toarray(), expected, rtol=RTOL)

    # ignore pow: needs square arrays
    for op in ARITHMETIC_OPERATORS:
        yield check_join_op, op


def test_join_ops_same_array():
    """
    array [op] self
    """

    def check_join_op(op):
        A = sdb.random((5, 5))
        C = op(A, A)
        assert_allclose(C.toarray(), op(A.toarray(), A.toarray()), rtol=RTOL)

    for op in ARITHMETIC_OPERATORS:
        yield check_join_op, op


def test_array_broadcast():
    """
    array [op] array2

    broadcastable shapes
    """
    def check_array_broadcast(shape1, shape2):
        A = sdb.random(shape1)
        B = sdb.random(shape2)
        C = A + B
        assert_allclose(C.toarray(), A.toarray() + B.toarray())

    for shapes in [((5, 1), 4), (4, (5, 1)),
                   ((5, 1), (1, 4)), ((1, 4), (5, 1)),
                   ((5, 1), (5, 5)), ((5, 5), (5, 1)),
                   ((1, 5, 1), 4), (4, (1, 5, 1)),
                   ((5, 1, 3), (4, 1)), ((4, 1), (5, 1, 3))]:
        yield check_array_broadcast, shapes[0], shapes[1]
