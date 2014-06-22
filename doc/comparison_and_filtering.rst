Comparison and Filtering
========================

SciDB-Py provides support for comparing and filtering
SciDB arrays, using NumPy-like syntax.

The standard inequality operators perform element-wise inequality testing between SciDB arrays, NumPy arrays,
and scalars::

    In [1]: from scidbpy import connect

    In [2]: sdb = connect()

    In [3]: x = sdb.arange(5)

    In [4]: x.toarray()
    Out[4]: array([0, 1, 2, 3, 4])

    In [5]: (x < 2).toarray()
    Out[5]: array([ True,  True, False, False, False], dtype=bool)

    In [6]: y = np.array([5, 0, 3, 2, 1])

    In [7]: (x < y).toarray()
    Out[7]: array([ True, False,  True, False, False], dtype=bool)

    In [13]: z = sdb.from_array(y)

    In [14]: (x < z).toarray()
    Out[14]: array([ True, False,  True, False, False], dtype=bool)

Array broadcasting is not currently performed when comparing two arrays -- they must have identical shapes.

As with NumPy arrays, boolean SciDB arrays can be used as masks, building a 1D array
whose elements correspond to the True locations in the mask::

    In [17]: r = sdb.random((3, 4))

    In [18]: r.toarray()
    Out[18]:
    array([[ 0.05799351,  0.84173237,  0.96347301,  0.34572826],
           [ 0.94962912,  0.30928932,  0.74852295,  0.0920418 ],
           [ 0.88262646,  0.49973551,  0.93622554,  0.31813278]])

    In [19]: r[r > .5].toarray()
    Out[19]:
    array([ 0.84173237,  0.96347301,  0.94962912,  0.74852295,  0.88262646,
            0.93622554])


Aggregation based on masked values
----------------------------------
A future version of SciDB-Py will provide a groupby operator, allowing comparisons
to be used to compute group-wise aggregates::

    sdb.groupby(x, x < 0.5).sum()

Until that method is added, you can perform the same computation with
two aggregate calls::

    mask = x < 0.5
    x[mask].sum()
    x[~mask].sum()

