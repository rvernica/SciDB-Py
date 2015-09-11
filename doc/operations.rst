
Basic Math on SciDB array objects
---------------------------------
.. currentmodule:: scidbpy

Operations on :class:`SciDBArray` objects generally return new
:class:`SciDBArray` objects.
The general idea is to promote function composition involving
:class:`SciDBArray` objects without moving data between SciDB and Python.

The ``scidbpy`` package provides quite a few common operations including
subsetting, pointwise application of scalar functions, aggregations, and
pointwise and matrix arithmetic.

Standard numpy attributes like ``shape``, ``ndim`` and ``size`` are defined for
:class:`SciDBArray` objects::

    >>> X = sdb.random((5, 10))
    >>> X.shape
    (5, 10)
    >>> X.size
    50
    >>> X.ndim
    2

Many SciDB-specific attributes are also defined,
including ``chunk_size``, ``chunk_overlap``, and ``sdbtype``, ::

    >>> X.chunk_size
    [1000, 1000]
    >>> X.chunk_overlap
    [0, 0]
    >>> X.sdbtype
    sdbtype('<f0:double>')

SciDBArrays also contain a ``datashape`` object, which encapsulates much of the
interface between Python and SciDB data, including the full array schema::

    >>> Xds = X.datashape
    >>> Xds.schema
    '<f0:double> [i0=0:4,1000,0,i1=0:9,1000,0]'


Scalar functions of SciDBArray objects (aggregations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: scidbpy.scidbarray

The package exposes the following aggregations:

============================ ==============================================
Name                         Description
============================ ==============================================
:meth:`~SciDBArray.min`      minimum value
:meth:`~SciDBArray.max`      maximum value
:meth:`~SciDBArray.sum`      sum of values
:meth:`~SciDBArray.var`      variance of values
:meth:`~SciDBArray.stdev`    standard deviation of values
:meth:`~SciDBArray.std`      standard deviation of values
:meth:`~SciDBArray.avg`      average/mean of values
:meth:`~SciDBArray.mean`     average/mean of values
:meth:`~SciDBArray.count`    count of nonempty cells
:meth:`~SciDBArray.approxdc` fast estimate of the number of distinct values
============================ ==============================================

**Examples: Minimum Aggregates**

Each operation can be computed across the entire array, or across specified
dimensions by passing the index or indices of the desired dimensions.
For example::

    >>> np.random.seed(0)
    >>> X = sdb.from_array(np.random.random((5, 3)))
    >>> X.toarray()
    array([[ 0.5488135 ,  0.71518937,  0.60276338],
           [ 0.54488318,  0.4236548 ,  0.64589411],
           [ 0.43758721,  0.891773  ,  0.96366276],
           [ 0.38344152,  0.79172504,  0.52889492],
           [ 0.56804456,  0.92559664,  0.07103606]])

Here we'll find the minimum of all values in the array.  The returned result
is a new SciDBArray, so we select the first element::

    >>> X.min()[0]
    0.071036058197886942

Like numpy, passing index 0 gives us the minimum within every column::

    >>> X.min(0).toarray()
    array([ 0.38344152,  0.4236548 ,  0.07103606])

Passing index 1 gives us the minimum within every row::

    >>> X.min(1).toarray()
    array([ 0.5488135 ,  0.4236548 ,  0.43758721,  0.38344152,  0.07103606])

Note that the convention for specifying aggregate indices here is designed
to match numpy, and is *opposite the convention used within SciDB*.
To recover SciDB-style aggregates, you can use the ``scidb_syntax`` flag::

    >>> X.min(1, scidb_syntax=True).toarray()
    array([ 0.38344152,  0.4236548 ,  0.07103606])

**Further Examples**

These operations return new :class:`SciDBArray` objects consisting of
scalar values.  Here are a few examples that materialize their results to
Python::

    >>> tridiag.count()[0]
    28
    >>> tridiag.sum()[0]
    20.0
    >>> tridiag.var()[0]
    1.6190476190476193

Note that a count of nonempty cells is also directly available from the
:func:`nonempty` function::

    >>> tridiag.nonempty()
    28

A related function is :func:`nonnull`, which counts the number of nonempty
cells which do not contain a null value.  In this case, the result is the
same as :func:`nonempty`::

    >>> tridiag.nonnull()
    28


Pointwise application of scalar functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The package exposes SciDB scalar-valued scalar functions that can be applied
element-wise to SciDB arrays:

==============     ==============================================
Function           Description
==============     ==============================================
:func:`sin`        Trigonometric sine
:func:`asin`       Trigonometric arc-sine / inverse sine
:func:`cos`        Trigonometric cosine
:func:`acos`       Trigonometric arc-cosine / inverse cosine
:func:`tan`        Trigonometric tangent
:func:`atan`       Trigonometric arc-tangent / inverse tagent
:func:`exp`        Natural exponent
:func:`log`        Natural logarithm
:func:`log10`      Base-10 logarithm
:func:`sqrt`       Square root
:func:`ceil`       Ceiling function
:func:`floor`      Floor function
:func:`is_nan`      Test for NaN values
==============     ==============================================

All trigonometric functions assume arguments are given in radians.
Here is a simple example that compares a computation in SciDB with
a local one (using the 'tridiag` array defined in the last examples)::

    >>> sin_tri = sdb.sin(tridiag)
    >>> np.linalg.norm(sin_tri.toarray() - np.sin(tridiag.toarray()))
    0.0


Shape and layout functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Arrays may be transposed and their data re-arranged into new shapes
with the usual :meth:`~SciDBArray.transpose` and :meth:`~SciDBArray.reshape`
functions::

    >>> tri_reshape = tridiag.reshape((20,5))
    >>> tri_reshape.shape
    (20, 5)
    >>> tri_reshape.transpose().shape
    (5, 20)
    >>> tri_reshape.T.shape  # shortcut for transpose
    (5, 20)


Arithmetic
^^^^^^^^^^

The package defines elementwise operations on all arrays and linear algebra
operations on matrices and vectors. Scalar multiplication is supported.

Element-wise sums and products::

    >>> np.random.seed(1)
    >>> X = sdb.from_array(np.random.random((10, 10)))
    >>> Y = sdb.from_array(np.random.random((10, 10)))
    >>> S = X + Y
    >>> D = X - Y
    >>> M = 2 * X
    >>> (S + D - M).sum()[0]
    -1.1102230246251565e-16

We can combine operations as well::

    >>> Z = 0.5 * (X + X.T)

There are also linear algebra operations (matrix-matrix product, matrix-vector
product) using the :func:`dot` function::

    >>> XY = sdb.dot(X, Y)
    >>> XY1 = sdb.dot(X, Y[:,1])
    >>> XTX = sdb.dot(X.T, X)


Broadcasting
^^^^^^^^^^^^

Numpy broadcasting conventions are generally followed in operations involving
differently-sized :class:`SciDBArray` objects. Consider the following example
that centers a matrix by subtracting its column average from each column.

First we create a test array with 5 columns::

    >>> np.random.seed(0)
    >>> X = sdb.from_array(np.random.random((10, 5)))

Now create a vector of column means::

    >>> xcolmean = X.mean(0)
    >>> xcolmean.shape
    (5,)

Subtract these means from the columns -- this is a broadcasting operation::

    >>> XC = X - xcolmean

To check that the columns are now centered,
we compute the column mean of ``XC``::

    >>> XC.mean(1).toarray()
    array([ -2.22044605e-17,   4.44089210e-17,  -1.11022302e-17,
             1.11022302e-16,  -3.33066907e-17])

The broadcasting operation which creates ``XC`` is implemented using a
join operation along dimension 1.


Lazy Evaluation
^^^^^^^^^^^^^^^
.. _lazy:

When possible, SciDB-Py defers actual database computation
until data are needed. It does this by using **lazy arrays**, which are
references to  as-yet unevaluated SciDB queries. Many array methods
actually return lazy arrays::

   >>> x = sdb.random((3,4))
   >>> x.name  # an array in the database
   'py1102522658694_00001'
   >>> y = x.mean(0)
   >>> y.name  # not yet in the database
   'aggregate(py1102522658694_00001,avg(f0),i1)'

Note that y's name doesn't refer to an array in the database, but
rather a query on x. Lazy arrays can also be identified by their non-null
`query` attribute::

   >>> y.query
   'aggregate(py1102522658694_00001,avg(f0),i1)'
   >>> x.query is None
   True

Calling :meth:`~SciDBArray.eval` forces lazy-arrays to be evaluated (it has no effect
on non-lazy arrays)::

   >>> y.eval()
   >>> y.name
   'py1102522658694_00014'

In most cases you don't need to worry about whether an array is lazy or not --
lazy arrays have all the same methods as regular arrays, and normally the
difference is transparent to the user. However, lazy arrays can be more
efficient with regard to compound queries. Consider an equation like
the law of cosines::

  c2 = a ** 2 + b ** 2 - 2 * a * b * sdb.cos(C)

This equation involves creating 7 intermediate data products:

 * ``t1 = a ** 2``
 * ``t2 = b ** 2``
 * ``t3 = 2 * a``
 * ``t4 = t3 * b``
 * ``t5 = sdb.cos(C)``
 * ``t6 = t4 * t5``
 * ``t7 = t1 + t2``
 * ``c2 = t7 - t6``

If ``a``, ``b``, and ``C`` are large SciDBArrays, this involves many
round-trip communiciations to the databse, several passes over the data,
and the storage of 7 arrays. Lazy arrays reduce this overhead by representing
some of these temporary arrays as unevaluated sub-queries. Passing larger
queries to SciDB at once also gives the database more opportunity to optimize
the final query, performing the computation in fewer passes over the data.

In some situations it's necessary or more efficient to
force evaluation of lazy arrays (often places where an array appears
several times in a complex query). Some SciDB-Py methods perform this
evaluation internally. You should also consider calling :meth:`~SciDBArray.eval` on lazy
arrays if you think the unevaluated queries are becoming too cumbersome.

