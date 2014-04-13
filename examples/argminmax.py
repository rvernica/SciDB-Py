"""
This shows an example of using SciDB-py to compute the argmin/argmax along
rows of a multi-dimensional array.

The queries are not optimized (some temporary arrays are used) but it shows
how the cross_join, filter, and redimension operators can be used in
conjunction to compute the desired result.
"""


def argmin(X, i):
    """
    Return a SciDBArray containing the column index of the minimum value
    along each row.

    This is functionally equivalent to numpy.argmin(X, i).

    Currently, this is implemented only for 2-dimensional X and i=1.
    """
    if i < 0:
        i += X.shape
    if not X.ndim == 2:
        raise NotImplementedError()
    if i != 1:
        raise NotImplementedError()

    # For efficiency, these queries should be combined
    X_min = X.min(0)
    J = sdb.cross_join(X, X_min, (0, 0))
    arr = sdb.new_array()
    sdb.query("store("
              "  redimension("
              "    filter({J}, {J.a0}={J.a1}),"
              "    <{J.d1}:int64>[{J.d0}=0:{N},{chunk},0]),"
              "  {arr})",
              J=J, arr=arr,
              N=J.shape[0] - 1,
              chunk=J.datashape.chunk_size[0])
    return arr


def argmax(X, i=-1):
    """
    Return a SciDBArray containing the column index of the maximum value
    along each row.

    This is functionally equivalent to numpy.argmax(X, i).

    Currently, this is implemented only for 2-dimensional X and i=1.
    """
    if i < 0:
        i += X.shape
    if not X.ndim == 2:
        raise NotImplementedError()
    if i != 1:
        raise NotImplementedError()

    # For efficiency, these queries should be combined
    X_max = X.max(0)
    J = sdb.cross_join(X, X_max, (0, 0))
    arr = sdb.new_array()
    sdb.query("store("
              "  redimension("
              "    filter({J}, {J.a0}={J.a1}),"
              "    <{J.d1}:int64>[{J.d0}=0:{N},{chunk},0]),"
              "  {arr})",
              J=J, arr=arr,
              N=J.shape[0] - 1,
              chunk=J.datashape.chunk_size[0])
    return arr

if __name__ == '__main__':
    from scidbpy import interface
    sdb = interface.SciDBShimInterface('http://vega.cs.washington.edu:8080')

    X = sdb.random((10, 5))
    print "numpy argmin:", X.toarray().argmin(1)
    print "scidb argmin:", argmin(X, 1).toarray()
    print
    print "numpy argmax:", X.toarray().argmax(1)
    print "scidb argmax:", argmax(X, 1).toarray()
