.. currentmodule:: scidbpy

.. _query:

The SciDB Query Interface
=========================

``scidbpy`` provides python wrappers for many useful SciDB operations, but
the SciDB AFL query language can provide even more customization
of operations (For more information on SciDB's AFL and AQL languages,
see the `SciDB Manual`_).  The :meth:`~SciDBInterface.query` function provides
a useful interface for generating raw queries by exploiting Python's
`String Formatting`_ syntax.  Through automatic insertion of the server-side
identifiers of SciDB arrays, attributes, and dimensions, the query interface
makes constructing complicated queries very convenient.

The general approach first creates a new :class:`SciDBArray` object and then
issues a query to populate data.  For example, to build an array of zeros
similar to the result of the :meth:`~SciDBInterface.zeros` function shown
above, the query can be constructed in the following way::

    >>> # first define an empty array to hold the result
    >>> zeros = sdb.new_array(shape=(5, 5), dtype='double')
    >>> # now execute a query to fill the array
    >>> sdb.query('store(build({A}, 0), {A})', A=zeros)
    >>> zeros.toarray()
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

The result is that ``zeros`` is a 10x10 array filled with zeros.  Here the
format statement ``{A}`` is replaced by the name of the desired array on
the SciDB server.

We can use this interface to quickly build more complex arrays.  For
example, to create an identity matrix similar to the result of the
:meth:`~SciDBInterface.identity` function shown above, we add a boolean check::

    >>> ident = sdb.new_array((5, 5), dtype='double')
    >>> sdb.query('store(build({A}, iif({A.d0}={A.d1}, 1, 0)), {A})',
    ...           A=ident)
    >>> ident.toarray()
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.]])


Here the substitutions ``{A.d0}`` and ``{A.d1}`` are replaced by the first
and second dimension names of the array referenced by ``A``.

Things can become even more complicated. The following example creates a
5x5 tridiagonal array, similar to the one used in the above examples::

    >>> tridiag = sdb.new_array((5, 5))
    >>> sdb.query('store(build({A}, \
    ...               iif({A.d0}={A.d1}, 2, iif({A.d0} <= {A.d1}+1 and {A.d0} >= {A.d1}-1, -1, 0))), {A})', A=tridiag)
    >>> tridiag.toarray()
    array([[ 2., -1.,  0.,  0.,  0.],
           [-1.,  2., -1.,  0.,  0.],
           [ 0., -1.,  2., -1.,  0.],
           [ 0.,  0., -1.,  2., -1.],
           [ 0.,  0.,  0., -1.,  2.]])

The query builds a tridiagonal array with 2 on the diagonal and -1 on
the sub- and super-diagonals. This shows how the query-formatting syntax
provided by the scidbpy package can be used to generate extremely powerful
AFL queries.

The full replacement syntax is outlined in the documentation of the
:meth:`~SciDBInterface.query` function.  It is a useful way to help
streamline the process of writing SciDB queries if and when it becomes
necessary.


.. _using_afl:

Working with the Array Functional Language
------------------------------------------

In addtion to the :class:`SciDBArray` class and ``query`` interface,
SciDB-Py provides a :ref:`direct binding <afl>` to SciDB's Array Functional Language, or AFL_.

The AFL consists of approximately
100 functions to perform array analysis. You can access these operators
through the ``afl`` attribute of a SciDB instance. More information on these operators can be found
on the `SciDB documentation <http://scidb.org/HTMLmanual/14.3/scidb_ug/ch17.html>`_.

AFL operators accept and return :class:`SciDBArray` objects, and thus
can be used interchangeably with methods on the :class:`SciDBArray` class itself.

Usage Example
^^^^^^^^^^^^^

::

    >>> from scidbpy import connect
    >>> sdb = connect()
    >>> x = sdb.random((3, 4))
    >>> afl = sdb.afl
    >>> # the number of elements greater than 0.5
    >>> cts = afl.aggregate(afl.filter(x, 'f0 > 0.5'), 'count(*)')
    >>> cts.query
    'aggregate(filter(py1100918363281_00001,f0 > 0.5),count(*))'
    >>> cts.toarray()
    array([8], dtype=uint64)

Note that each AFL operator includes documentation from the official
SciDB manual::

    In [37]: afl.filter?
    Type:        function
    String form: <function filter at 0x104eb1c08>
    File:        /Users/beaumont/scidbpy/scidbpy/afl.py
    Definition:  afl.filter(*args)
    Docstring:
    filter( srcArray, expression )

    Produces a result array by filtering out (mark as empty) the cells in the source array     for which the expression evaluates to False.

    Parameters
    ----------

        - srcArray: a source array with srcAttrs and srcDims.
        - expression: an expression which takes a cell in the source array
          as input and evaluates to either True or False.

.. note::

    Some AFL functions, like :func:`list`, expect single-quoted strings
    as input. These quotes must be explicitly provided (e.g. ``afl.list("'arrays'")``).

Chaining AFL Operators
^^^^^^^^^^^^^^^^^^^^^^
.. _afl_chain:

Many SciDB queries involve nesting several AFL calls, with
the result of an inner call included as the first argument of an outer call.
For example::

    afl.filter(afl.project(afl.apply(x, 'y', 'f0+1'), 'y'), 'y > 3')

SciDB-py includes some syntactic sugar for building queries like this: for
AFL operators whose names don't collide with another :class:`SciDBArray` method,
``x.operator(...)`` is equivalent to ``afl.operator(x, ...)``. Thus, the above
query can be re-written as::

    x.apply('y', 'f0+1').project('y').filter('y > 3')

These two syntaxes are equivalent.

.. warning::

    This syntax only applies to AFL operators that don't collide
    with a :class:`SciDBArray` method name.

.. _SciDB: http://scidb.org/

.. _AFL: http://scidb.org/HTMLmanual/14.3/scidb_ug/ch17.html

.. _`SciDB Manual`: http://www.scidb.org/HTMLmanual/

.. _`String Formatting`: http://docs.python.org/2/whatsnew/2.6.html#pep-3101-advanced-string-formatting
