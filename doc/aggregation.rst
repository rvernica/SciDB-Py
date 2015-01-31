Aggregation and Join Operations
================================
.. currentmodule:: scidbpy

SciDB-Py provides several high-level functions to perform
database-style joins and aggregations. The syntax of these
functions are modeled after Pandas.


Groupby
-------

The :meth:`~SciDBArray.groupby` operation allows you to partition an array into groups based on the value in one or more columns, and then perform
operations on each group separately::

    In [2]: import pandas as pd
    In [3]: df = pd.DataFrame({'x': [1, 1, 1, 2, 2, 1, 3], 'y':[1, 2, 3, 4, 5, 6, 7]})
    In [4]: x = sdb.from_dataframe(df)
    In [5]: x.groupby('x').aggregate('sum(y)').todataframe()
    Out[5]:
         x_cat  y_sum  x
    idx
    0        0     12  1
    1        1      9  2
    2        2      7  3

:meth:`~SciDBArray.groupby` takes one or more attribute or dimension names
as inputs, and returns an intermediate :class:`~scidbpy.aggregation.GroupBy`
object. Calling :meth:`~scidbpy.aggregation.GroupBy.aggregate` on this object aggregates over groups.

The argument to groupby can be either:

 * A string, interpreted as a SciDB aggregate command
 * A dict, mapping output attribute names to SciDB Aggregation commands

For example::

    In [17]: x.groupby('x').aggregate({'y_sum':'sum(y)', 'y_max':'max(y)'}).todataframe()
    Out[17]:
         x_cat  y_sum  y_max  x
    idx
    0        0     12      6  1
    1        1      9      5  2
    2        2      7      7  3

Grouping on attributes
^^^^^^^^^^^^^^^^^^^^^^
When attribute names are used to group arrays, they are first lexicographically sorted and converted into categorical dimensions. This
kind of grouping is more expensive than a grouping on dimension names.


Aggregate
---------
The :meth:`~SciDBArray.aggregate` method behaves similarly to GroupBy,
but provides a syntax more akin to R. It takes a `by` argument to support
grouping on dimension values. However, unlike GroupBy, the names passed to `by` *must* be dimensions

Database-style joins
---------------------
The :meth:`~scidbpy.interface.SciDBInterface.merge` method mimics the Pandas merge function,
to perform database-style joins on two arrays. When joining two arrays,
they are first aligned along common values of one or more *join dimensions*.
Then, the attributes of each array are concatenated.

Like GroupBy, :meth:`~scidbpy.interface.SciDBInterface.merge` automatically computes
categorical dimensions to support joining on attribute names::

    In [34]: x = sdb.arange(5)
    In [35]: x['f1'] = 'f0 * 10'
    In [36]: y = sdb.arange(6) * 3
    In [37]: x.todataframe()
    Out[37]:
        f0  f1
    i0
    0    0   0
    1    1  10
    2    2  20
    3    3  30
    4    4  40

    In [44]: y.todataframe()
    Out[44]:
         x
    i0
    0    0
    1    3
    2    6
    3    9
    4   12
    5   15

    In [45]: sdb.merge(x, y).todataframe()
    Out[45]:
        f0  f1   x
    i0
    0    0   0   0
    1    1  10   3
    2    2  20   6
    3    3  30   9
    4    4  40  12

    In [46]: sdb.merge(x, y, left_on='f0', right_on='x').todataframe()
    Out[46]:
                      f0  f1  x
    i0_x f0_cat i0_y
    0    0      0      0   0  0
    3    3      1      3  30  3


.. note::
   Merges are currently restricted to inner joins

.. note::
   Prior to SciDB-Py v14.10, the `merge` function performed a direct AFL
   merge() call. It now performs the higher-level function described above.


