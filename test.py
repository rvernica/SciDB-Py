from scidbpy.interface import SciDBShimInterface
import time

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

    x = sdb.ones(100)
    print sdb.list_arrays()
    del x
    print sdb.list_arrays()

test2()
