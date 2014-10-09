.. _comparison_and_filtering:


Comparing and Filtering Arrays
==============================

.. currentmodule:: scidbpy

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

As with NumPy arrays, boolean SciDB arrays can be used as masks::

    In [4]: r = sdb.random((3,4))

    In [5]: r.toarray()
    Out[5]:
    array([[ 0.72039148,  0.6497302 ,  0.84122248,  0.87304017],
           [ 0.14896572,  0.71237498,  0.21999935,  0.14793879],
           [ 0.69345283,  0.18611741,  0.43660223,  0.06478555]])

    In [6]: r[r > 0.5].toarray()
    Out[6]:
    array([[ 0.72039148,  0.6497302 ,  0.84122248,  0.87304017],
           [ 0.        ,  0.71237498,  0.        ,  0.        ],
           [ 0.69345283,  0.        ,  0.        ,  0.        ]])


Note that this masking behavior is different than NumPy -- NumPy collapses
the input array when masking, returning a 1D result of unmasked items.
To reproduce this behavior in SciDB-Py, use the :meth:`~SciDBArray.collapse` method::

    In [9]: r[r > 0.5].collapse().toarray()
    Out[9]:
    array([ 0.72039148,  0.6497302 ,  0.84122248,  0.87304017,  0.71237498,
            0.69345283])

SciDB-Py behaves this way in order to retain the location of unmasked items,
which is often useful information. For example, we can see these locations
when using :meth:`~SciDBArray.todataframe`::

    In [10]: r[r > 0.5].todataframe()
    Out[10]:
                 f0
    i0 i1
    0  0   0.720391
       1   0.649730
       2   0.841222
       3   0.873040
    1  1   0.712375
    2  0   0.693453

.. note:: SciDB-Py's masking behavior was changed in version 12.10. Prior
          to this, SciDB-Py collapsed results like NumPy

Extracting values along a particular axis
-----------------------------------------

Use the :meth:`SciDBArray.compress` method to extract row or column subsets of an array. For example, to extract all rows where
the sum across all columns exceeds a threshold::

   In [3]: x = sdb.random((3,4))

   In [4]: x.toarray()
   Out[4]:
   array([[ 0.10111977,  0.5511177 ,  0.49532397,  0.4213646 ],
          [ 0.3812068 ,  0.97679566,  0.20473656,  0.40256096],
          [ 0.2387294 ,  0.88714084,  0.01064819,  0.48275173]])

   In [5]: x.mean(1).toarray()
   Out[5]: array([ 0.39223151,  0.491325  ,  0.40481754])

   In [6]: x.compress(x.mean(1) > 0.4, axis=0).toarray()
   Out[6]:
   array([[ 0.3812068 ,  0.97679566,  0.20473656,  0.40256096],
          [ 0.2387294 ,  0.88714084,  0.01064819,  0.48275173]])


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

Comparison with SciDB-R
-----------------------
See `this page <https://github.com/Paradigm4/SciDBR/wiki/Comparing-and-filtering-values>`_ for SciDB-R's syntax for comparing and filtering arrays.

