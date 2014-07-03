
.. currentmodule:: scidbpy

.. _creation:

Creating arrays
---------------
.. Download example code :download:`here<creation.py>`:

The following sections illustrate a number of ways to create :class:`SciDBArray`
objects. The examples assume that an ``sdb`` interface object has already
been set up.

From a numpy array
^^^^^^^^^^^^^^^^^^

Perhaps the simplest approach to creating an arbitrary :class:`SciDBArray`
object is to upload a numpy array into SciDB with the
:meth:`~SciDBInterface.from_array`
function. Although this approach is very convenient, it is not really suitable
for very big arrays (which might exceed memory availability in a single
computer, for example). In such cases, consider other options described below.

The following example creates a :class:`SciDBArray` object named ``Xsdb``
from a small 5x4 numpy array named ``X``.

.. literalinclude:: creation.py
   :lines: 6-10

The package takes care of naming the SciDB array in this example (use
``Xsdb.name`` to see the SciDB array name).

From a scipy sparse matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^

In a similar way, a :class:`SciDBArray` can be created from a scipy sparse
matrix.  For example:

.. literalinclude:: creation.py
   :lines: 13-17

This operation is most efficient for matrices stored in coordinate form
(``coo_matrix``).  Other sparse formats will be internally converted to
COO form in the process of transferring the data.


Convenience array creation functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many standard numpy functions for creating special arrays are supported.  These
include:

:meth:`~SciDBInterface.zeros`
    to create an array full of zeros:

.. literalinclude:: creation.py
   :lines: 19-20

:meth:`~SciDBInterface.ones`
    to create an array full of ones:

.. literalinclude:: creation.py
   :lines: 21-22

:meth:`~SciDBInterface.random`
    to create an array of uniformly distributed random floating-point values:

.. literalinclude:: creation.py
   :lines: 24-26

:meth:`~SciDBInterface.randint`
    to create an array of uniformly distributed random integers:

.. literalinclude:: creation.py
   :lines: 28-30

:meth:`~SciDBInterface.arange`
    to create and array with evenly-spaced values given a step size:

.. literalinclude:: creation.py
   :lines: 32-33

:meth:`~SciDBInterface.linspace`
    to create an array with evenly spaced values between supplied bounds:

.. literalinclude:: creation.py
   :lines: 35-37

:meth:`~SciDBInterface.identity`
    to create a sparse or dense identity matrix:

.. literalinclude:: creation.py
   :lines: 39-40

These functions should be familiar to anyone who has used NumPy, and the
syntax of each function closely follows its NumPy counterpart.  In each case,
the array is defined and created directly in the SciDB server, and the
resulting Python object is simply a wrapper of the native SciDB array.
Because of this, the functions outlined here and in the following sections
can be more efficient ways to generate large SciDB arrays than copying data
from a numpy array.

.. note:: SciDB does not yet have a way to set a random seed, prohibiting
          reproducible results involving the random number generator.


From an existing SciDB array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, :class:`SciDBArray` objects may be created from existing SciDB arrays, so
long as the data type restrictions outlined above are met. (It usually makes
sense to load large data sets into SciDB externally from the Python package,
using the SciDB parallel bulk loader or similar facility.)

The following example uses the :meth:`~SciDBArray.query` function to build
and store a small 10x5 SciDB array named "A" independently of Python.
We then create a :class:`SciDBArray`
object from the SciDB array with the :meth:`~SciDBInterface.wrap_array` function, passing
the name of the array identifier on the SciDB server:

.. literalinclude:: creation.py
   :lines: 42-50

Note that there are some restrictions on the types of arrays which can be
wrapped by ``SciDB-Py``.  The array data must be of a compatible type, and
have integer indices.  Also, arrays with indices that don't start at zero
may not behave as expected for item access and slicing, discussed below.

Note also that many functions in the SciDB-Py package work on single-attribute
arrays. When a :class:`SciDBArray` object refers to a SciDB array with more
than one attribute, only the first listed attribute is used.


Persistence of SciDB-Py arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each array has a :attr:`~SciDBArray.persistent` attribute.
When ``persistent`` is set to ``True``, arrays remain in SciDB
until explicitly removed by a ``remove`` query.
If ``persistent`` is set to False, the arrays are removed when the
:meth:`SciDBInterface.reap` or :meth:`SciDBArray.reap` methods are invoked.
(Note that :meth:`interface.SciDBInterface.reap` is automatically invoked when
Python exits).

Arrays defined from an existing SciDB array using the
:meth:`~interface.SciDBInterface.wrap_array` argument are always persistent, while
all other array creation routines set ``persistent=False`` by default:

.. literalinclude:: creation.py
    :lines: 52-55

When :func:`~interface.connect` is used as a context manager, non-persistent
arrays are reaped at the end of the context block:


.. literalinclude:: creation.py
    :lines: 57-59
