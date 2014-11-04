.. _access:

Accessing array data
====================

Converting arrays to other data structures
------------------------------------------

.. currentmodule:: scidbpy

SciDB-Py is designed to perform operations on SciDB arrays in a
natural Python dialect, computing those operations in SciDB while minimizing data traffic between the database and Python. However, it is useful to materialize SciDB array data to Python, for example to obtain and plot results.

:class:`SciDBArray` objects provide several functions that materialize array
data to Python:

:meth:`~SciDBArray.toarray`
    can be used to populate a ``numpy`` array from an
    `N`-dimensional array with any number of attributes::

        >>> A = sdb.linspace(0, 10, 5)
        >>> A.toarray()
        array([  0. ,   2.5,   5. ,   7.5,  10. ])

        >>> B = sdb.join(sdb.linspace(0, 8, 5), sdb.arange(5, dtype=int))
        >>> B.toarray()
        array([(0.0, 0), (2.0, 1), (4.0, 2), (6.0, 3), (8.0, 4)],
                  dtype=[('f0', '<f8'), ('f0_2', '<i8')])

:meth:`~SciDBArray.tosparse`
    can be used to populate a `SciPy sparse matrix`_ from a 2-dimensional
    array with a single attribute::

        >>> I = sdb.identity(5, sparse=True)
    >>> I.tosparse(sparse_fmt='dia')
        <5x5 sparse matrix of type '<type 'numpy.float64'>'
            with 5 stored elements (1 diagonals) in DIAgonal format>

    :meth:`~SciDBArray.tosparse` will also work with 1-dimensional arrays
    or multi-dimensional arrays; in this case the result cannot be exported
    to a SciPy sparse format, but will be returned as a
    `Numpy record array`_ listing the indices and values.


:meth:`~SciDBArray.todataframe`
    can be used to populate a `Pandas dataframe`_ from a 1-dimensional
    array with any number of attributes::

        >>> B = sdb.join(sdb.linspace(0, 8, 5, dtype='<A:double>'),
                         sdb.arange(1, 6, dtype='<B:int32>'),
                         sdb.ones(5, dtype='<C:float>'))
        >>> B.todataframe()
           A  B  C
        0  0  1  1
        1  2  2  1
        2  4  3  1
        3  6  4  1
        4  8  5  1

These methods are discussed in greater detail in :ref:`downloading`.

Element Access
--------------

Single elements of :class:`SciDBArray` objects can be referenced with the
standard numpy indexing syntax.  These single elements are returned by
value::

   >>> x = sdb.arange(12).reshape((3,4))
   >>> x[1, 2]
   6

Note that element assignment (e.g. ``x[0, 0] = 4``)is not supported.

Subarrays and Slice Syntax
--------------------------

SciDBArrays support NumPy's slice syntax for extracting subregions::

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

Some of NumPy's "Fancy Indexing" operations, like indexing
with a boolean array, are also supported; see :ref:`comparison_and_filtering`.


.. _fancy_indexing:
You can also index arrays using integer arrays::

   >>> x = sdb.arange(100) * 5
   >>> y = sdb.from_array(np.array([ 3,  3,  5, 10, 30, 20,  5]))
   >>> x[y].toarray()
   array([ 15,  15,  25,  50, 150, 100,  25])



Slicing by dimension name
-------------------------
The :meth:`~SciDBArray.isel` method allows you to index into arrays
by dimension name instead of position::

    >>> x = sdb.arange(30).reshape((6, 5))
    >>> x.schema
    '<f0:int64> [i0=0:5,1000,0,i1=0:4,1000,0]'
    >>> x.isel(i1=2).toarray()   # same as x[:, 2]
    array([ 2,  7, 12, 17, 22, 27])


Attribute access
----------------
.. _attribute_access:

You can access specific attributes of an array by
passing their names in the brackets. You can also add new attributes
by providing a SciDB expression::

    >>> x = sdb.arange(4)
    >>> x.att_names
    ['f0']
    # extract the f0 attribute
    >>> x['f0'].toarray()
    array([0, 1, 2, 3])

    # add a new attribute, and access it
    >>> x['y'] = 'sin(f0 * 3)'
    >>> x['y'].toarray()
    array([ 0.        ,  0.14112001, -0.2794155 ,  0.41211849])

    # multi-attribute access
    >>> x[['y', 'f0']].toarray()
    array([(0.0, 0), (0.1411200080598672, 1), (-0.27941549819892586, 2),
          (0.4121184852417566, 3)],
         dtype=[('y', '<f8'), ('f0', '<i8')])


.. _`SciPy sparse matrix`: http://docs.scipy.org/doc/scipy/reference/sparse.html

.. _`Pandas dataframe`: http://pandas.pydata.org/pandas-docs/dev/dsintro.html#dataframe

.. _`Numpy record array`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html

