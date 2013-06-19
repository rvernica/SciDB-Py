from scidbpy.interface import SciDBShimInterface
import time
import numpy as np

def test():
    scidb = SciDBShimInterface('http://localhost:8080')
    # list all arrays
    print scidb._execute("list('arrays')", response=True)

    # build an array
    arrayname = scidb._create_array('<val:double>[i=0:9,10,0]')
    #execute("store(build(<val:double>[i=0:9,10,0],1), myarray)",
    #                    response=False)

    # list all arrays
    print scidb._execute("list('arrays')", response=True)

    # remove the array
    scidb._delete_array(arrayname)
    #print scidb.execute("remove(myarray)",
    #                    response=False)
    print scidb._execute("list('arrays')", response=True)

def test2():
    sdb = SciDBShimInterface('http://localhost:8080')

    x = sdb.random((10, 4))
    print sdb.list_arrays()
    print x.toarray()
    del x
    print sdb.list_arrays()

def test3():
    sdb = SciDBShimInterface('http://localhost:8080')
    A = sdb.random((4, 6))
    B = sdb.random((6, 5))
    C = sdb.dot(A, B)

    print "SciDB dot-product:"
    print C.toarray()
    print C.datashape.descr
    print C
    print
    print "Numpy dot-product:"
    print np.dot(A.toarray(), B.toarray())

def test4():
    sdb = SciDBShimInterface('http://localhost:8080')
    A = sdb.random((5, 10), chunk_size=32)

    U, S, VT = sdb.svd(A)
    print S.toarray()

    U2, S2, VT2 = np.linalg.svd(A.toarray())
    print S2

def test5():
    sdb = SciDBShimInterface('http://localhost:8080')
    arr_in = np.random.random((5, 5))
    A = sdb.from_array(arr_in)
    print arr_in
    print A.toarray()
    

test5()
