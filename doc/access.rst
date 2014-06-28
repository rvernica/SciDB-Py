.. _access::
.. currentmodule:: scidbpy.scidbarray

Accessing elements in a SciDBArray
==================================

SciDBArrays are containers for arrays in a SciDB database. To download an entire array for local access, use the :meth:`SciDBArray.toarray` or :meth:`SciDBArray.todataframe` methods. These convert the data into a NumPy array and Pandas DataFrame, respectively::

    >>> sdb = connect()
    >>> x = sdb.arange(5)
    >>> x_np = x.toarray()
    >>> x_df = x.todataframe()

    >>> print (x_np)
    [0, 1, 2, 3, 4]
    (array([0, 1, 2, 3, 4]), <type 'numpy.ndarray'>)
    >>> print(x_df)
          0
       0  0
       1  1
       2  2
       3  3
       4  4

Slice Syntax
------------

SciDBArrays support NumPy's slice syntax to extracting subregions::

    >>> x = sdb.arange(30).reshape((6, 5))
    >>> x.toarray()
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24],
           [25, 26, 27, 28, 29]])
    >>> x[0:2].toarray() # the first 2 rows
    array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
    >>> x[:, 1:3].toarray()  # the second 2 columns
    array([[ 1,  2],
           [ 6,  7],
           [11, 12],
           [16, 17],
           [21, 22],
           [26, 27]])
    >>> x[::2].toarray()  # every other row
    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14],
           [20, 21, 22, 23, 24]])


Attribute access
----------------
Similar to Pandas, you can access specific attributes of an array by
passing their names in the brackets. You can also add new attributes
to the data by providing a SciDB expression as a string::

    >>> x = sdb.arange(4)
    >>> x.att_names
    ['f0']
    >>> x['f0'].toarray()
    array([0, 1, 2, 3])
    >>> x['y'] = 'sin(f0 * 3)'
    >>> x['y'].toarray()
    array([ 0.        ,  0.14112001, -0.2794155 ,  0.41211849])
    >>> x[['y', 'f0']].toarray()
    array([(0.0, 0), (0.1411200080598672, 1), (-0.27941549819892586, 2),
          (0.4121184852417566, 3)],
         dtype=[('y', '<f8'), ('f0', '<i8')])

