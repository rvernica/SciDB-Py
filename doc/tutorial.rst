========
Tutorial
========

SciDB is an open-source database that organizes data in n-dimensional arrays.
SciDB features include ACID transactions, parallel processing, distributed
storage, efficient sparse array storage, and native parallel linear algebra
operations.  The scidbpy package for Python defines a numpy array-like
interface for SciDB arrays in Python.  The arrays mimic numpy arrays, but
operations on them are performed by the SciDB engine.  Data are materialized to
Python only when requested. A basic set of array subsetting, arithmetic and
utility operations are defined by the package. Additionally, a general query
function provides a mechanism for performing arbitrary queries on scidbpy array
objects.


Software prerequisites
----

The scidbpy package requires at least:

1. An available SciDB installation
2. The 'shim' SciDB network interface.

We assume an existing installation of SciDB is available. Binary SciDB packages
(for Ubuntu 12.04 and RHEL/CentOS 6) and source code are available from
http://scidb.org.  The examples in this tutorial assume that SciDB is running
on a computer with host name "localhost." If SciDB is not running on localhost,
adjust the name accordingly.

The scidbpy package requires installation of a simple HTTP network service
called "shim" on the computer that SciDB coordinator is installed on. The
network service only needs to be installed on the SciDB computer, not on client
computers that connect to SciDB from Python. It's available in packaged binary
form for supported SciDB operating systems, and as source code which can be
compiled and deployed on any SciDB installation.

See http://github.com/paradigm4/shim  for source code and installation
instructions.


Loading the scidbpy package and connecting to SciDB
----

The following example loads the package and defines an object named `sdb`
that represents the SciDB interface. The example assumes that the SciDB
coordinator is on the computer with host name 'localhost'--adjust the
host name as required if SciDB is on a different computer::

  import numpy as np
  from scidbpy import interface, SciDBQueryError, SciDBArray
  sdb = interface.SciDBShimInterface('http://localhost:8080')

The following examples refer to an interface object named `sdb` similar to
the illustration.

SciDB arrays 
----

SciDB arrays are composed of cells. Each cell may contain one or more values
referred to as attributes. The data types and number of attributes are
consistent across all cells within one array. All the attribute values within a
cell may be left undefined, in which case the cell is called empty. Arrays with
empty cells are referred to as sparse arrays in the SciDB documentation.

Individual attribute values may also be explicitly marked missing with one of
several possible SciDB null codes.

Cells are arranged by an integer coordinate system into n-dimensional arrays.
SciDB uses signed 64-bit integers for coordinates. Each coordinate axis is
typically referred to as a dimension in the SciDB documentation. SciDB is
limited in theory to about 100 dimensions, but in practice that limit is
typically much lower (up to say, 10 dimensions or so). The default SciDB
array origin is usually zero. But SciDB arrays  may use any signed 64-bit
integer origin.



Creating SciDB array objects
----

The `SciDBArray` class defines the primary method of interaction between Python
and SciDB. `SciDBArray` objects are Python representations of SciDB arrays that
mimic numpy arrays in may ways.  `SciDBArray` array objects are limited to the
following SciDB array attribute data types: bool, float32, float64, int8,
int16, int32, int64, uint8, unit16, uint32, uint64, and singe characters. 

The following sections illustrate a number of ways to create `SciDBArray`
objects. The examples assume that an `sdb` interface object has already
been set up.

From a numpy array
^^^^^

The simplest approach to creating a `SciDBarray` object is to upload a numpy
array into SciDB with the `from_array` function. Although this approach is
super-convenient, it is not really suitable for very big arrays (which might
exceed memory availability in a single computer, for example). In such cases,
consider other options described below.

The following example creates a `SciDBArray` object named `Xsdb`
from a small 5x4 numpy array named `X`::

  X = np.random.random((5, 4))
  Xsdb = sdb.from_array(X)

The package takes care of naming the SciDB array in this example (use
`Xsdb.name` to see the SciDB array name).


Convenience array creation functions
^^^^^

Many standard numpy functions for creating special arrays are supported.  These
include `zeros` to create an array full of zeros, `random` to create an array
of uniformly distributed random floating-point values, `randint` to create an
array of uniformly distributed random integers, and `arange` and `linspace` to
create arrays with evenly spaced values between supplied bounds. These functions
closely follow their numpy counterparts. The data in each case are defined by
SciDB.

The functions outlined here and in the following sections can be more efficient
ways to generate large SciDB arrays than copying data from a numpy array since
the data are generated at the SciDB back-end.

Examples follow::

  # Create a 10x10 array of double-precision zeros:
  A = sdb.zeros( (10,10) )

  # Create a 10x10 array of 64-bit signed integer ones:
  A = sdb.ones( (10,10), dtype='int64' )

  # Create a 10x10 array of numbers between -1 and 2 (inclusive) sampled from a uniform random distribution.
  A = sdb.random( (10,10), lower=-1, upper=2)

  # Create a vector of 5 equally spaced numbers between 1 and 10, including the endpoints:
  A = sdb.linspace(1,10,num=5,endpoint=True)

  # Create a 10x10 sparse, double-precision-valued identity matrix:
  A = sdb.identity(10, dtype='double', sparse=True)

Note: SciDB does not yet have a way to set a random seed, prohibiting
reproducible results involving the random number generator.


From a SciDB query
^^^^^

The SciDB query interface provides a simple way to create `SciDBArray` objects
from arbitrary SciDB queries using the SciDB AFL language. The `query` function
greatly simplifies the use of AFL by allowing references to `SciDBArray`
object in place of raw SciDB schema information in the query.

`SciDBArray` object references may be supplied in the query string using a
string replacement syntax that flexibly supplies SciDB schema, attribute and
dimension names to the query. The replacement syntax is fully outlined in the
documentation.

The general approach first creates a new `SciDBArray` object and then issues
a query to populate data. The following example creates a 10x10 sparse (in
the SciDB sense) tridiagonal array::

  arr = sdb.new_array((10, 10))
  sdb.query('store(build_sparse({A},iif({A.d0}={A.d1},2,{A.d1}-{A.d0}),{A.d0}<={A.d1}+1 and {A.d0}>={A.d1}-1), {A})', A=arr)


From an existing SciDB array
^^^^^

Finally, `SciDBArray` objects may be created from existing SciDB arrays, so
long as the data type restrictions outlined above are met. (It usually makes
sense to load large data sets into SciDB externally from the Python package,
using the SciDB parallel bulk loader or similar facility.)

The following example uses the `query` function to build and store a small 10x5
SciDB array named "A" independently of Python. We then create a `SciDBArray`
object from the SciDB array with the `new_array` function using the `scidbname`
argument::

  sdb.query("remove(A)")
  sdb.query("store(build(<v:double>[i=1:10,5,0,j=1:5,5,0],i+j),A)")
  A = sdb.new_array(scidbname="A")

Note that subarray indexing of `SciDBArray` objects follows numpy convention.
SciDB arrays with negative-valued coordinate indices don't fit this convention
and should be translated to a coordinate system with a nonnegative origin before
use.

Note that most functions in the scidbpy package work on single-attribute
arrays. When a `SciDBArray` object refers to a SciDB array with more than one
attribute, the first listed attribute is used.

Scope of scidbpy arrays
^^^^^

The `new_array` function takes an argument named `persistent`. When
`persistent` is set to True, arrays last in SciDB until explicitly removed.
Otherwise, arrays are removed from SciDB when they fall out of scope in the
Python session. Arrays defined from an existing SciDB array using the
`scidbname` argument are always persistent.


Retrieving data from SciDB array objects
----

A central idea of the package is to program operations on SciDB arrays in a
natural Python dialect, computing those operations in SciDB while minimizing
data traffic between SciDB and Python. However, it is useful to materialize
SciDB array data to Python, for example to obtain and plot results.

`SciDBArray` objects provide two functions that materialize array data to
Python. Use the `toarray` function to bring data back as a numpy array,
and the `tosparse` function to return data in a form compatible with
scipy sparse arrays.

Let's revisit an earler example and retrieve its data in dense and
sparse formats::

  # Build a sparse tridiagonal SciDB array:
  arr = sdb.new_array((10, 10))
  sdb.query('store(build_sparse({A},iif({A.d0}={A.d1},2,{A.d1}-{A.d0}),{A.d0}<={A.d1}+1 and {A.d0}>={A.d1}-1), {A})', A=arr)

  # Materialize SciDB array to Python as a numpy array:
  arr.toarray()
  # Produces output like:
  array([[ 2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [-1.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0., -1.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0., -1.,  2.,  1.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0., -1.,  2.,  1.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0., -1.,  2.,  1.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0., -1.,  2.,  1.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  2.,  1.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  2.,  1.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  2.]])

  # Materialize SciDB array to Python as a sparse array:
  from scipy import sparse
  arr.tosparse('csr')
  # Produces output like:
  <10x10 sparse matrix of type '<type 'numpy.float64'>'
        with 28 stored elements in Compressed Sparse Row format>


Operations on SciDB array objects
----

Operations on `SciDBArray` objects generally return new `SciDBArray` objects.
The general idea is to promote function composition involving `SciDBArray`
objects without moving data between SciDB and Python

The scidbpy package provides quite a few common operations including
subsetting, pointwise application of scalar functions, aggregations, and
pointwise and matrix arithmetic.

Standard numpy attributes like `shape`, `ndim` and `size` are defined for
`SciDBArray` objects. Many SciDB-specific attributes are also defined,
including `chunk_size`, `chunk_overlap`, and `sdbtype`.

Subarrays
^^^^

Rectilinear subarrays are selected with standard numpy syntax. Subarrays of
`SciDBArray` objects are new `SciDBArray` objects. Consider the
example from the last section::

  arr = sdb.new_array((10, 10))
  sdb.query('store(build_sparse({A},iif({A.d0}={A.d1},2,{A.d1}-{A.d0}),{A.d0}<={A.d1}+1 and {A.d0}>={A.d1}-1), {A})', A=arr)

  # Define a 3x10 subarray (returned as a new SciDBArray object)
  X = arr[2:5,:]
  X.toarray()
  array([[ 0., -1.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0., -1.,  2.,  1.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0., -1.,  2.,  1.,  0.,  0.,  0.,  0.]])

Fancy (tuple-based) subarray indexing is not yet supported.

Note that subarray indexing of `SciDBArray` objects follows numpy convention.
SciDB arrays with negative-valued coordinate indices should be translated to a
coordinate system with a nonnegative origin before use.

Scalar functions of SciDBArray objects (aggregations)
^^^^

The package exposes the following aggregations:

`min`, `max`, `sum`, `var` (variance), `stdev` or `std` (standard deviation),
`avg` or `mean`, `approxdc` (fast estimation of the number of distinct values)
and `count` (count of nonempty array cells).

These operations return new `SciDBArray` objects consisting of scalar values.
Here are a few examples that materialize their results to Python (using
the `arr` array defined in the last example)::

  sdb.count(arr).toarray()[0]
  28

  sdb.sum(arr).toarray()[0]
  20.0

  sdb.var(arr).toarray()[0]
  1.6190476190476193

Note that a count of nonempty cells is also directly available from the
`nonempty` function::

  arr.nonempty()
  28

Pointwise application of scalar functions
^^^^

The package exposes SciDB scalar-valued scalar functions that can be applied
element-wise to SciDB arrays: `sin, cos, tan, asin, acos, atan, exp, log` and
`log10`. Here is a simple example that compares a computation in SciDB with
a local one (using the `arr` array defined in the last examples)::

  np.linalg.norm(sdb.sin(arr).toarray() - np.sin(arr.toarray()))
  0.0


Shape and layout functions
^^^^

Arrays may be transposed and their data re-arranged into new shapes
with the usual `transpose` and `reshape` functions::

  arr.transpose()
  arr.reshape((20,5)).shape
  (20,5)


Arithmetic
^^^^

The package defines elementwise operations on all arrays and linear algebra
operations on matrices and vectors. Scalar multiplication is supported.

Element-wise sums and products::

  np.random.seed(1)
  X = sdb.from_array(np.random.random((10,10)))
  Y = sdb.from_array(np.random.random((10,10)))

  S = X + Y
  D = X - Y
  M = 2 * X
  (S + D - M).sum()[0]
  -1.1102230246251565e-16

  0.5*(X + X.T)


Linear algebra operations::

  sdb.dot(X,Y)
  sdb.dot(X,Y[:,1])
  sdb.dot(X.T, X)


Broadcasting
^^^^

Numpy broadcasting conventions are generally followed in operations involving
differently-sized `SciDBArray` objects. Consider the following example that
centers a matrix by subtracting its column average from each column::

  from scidbpy import interface, SciDBQueryError, SciDBArray
  sdb = interface.SciDBShimInterface('http://localhost:8080')
  import numpy as np

  # Create a small test array with 5 columns:
  X = sdb.from_array(np.random.random((10,5)))

  # Create a vector of column means:
  xcolmean = X.mean(1)

  # Subtract the column means from the original matrix using broadcasting:
  XC = X - xcolmean

The example populates the column mean values in the vector `xcolmean` by
issuing a SciDB aggregation query along the columns of X.


Example applications
----

Covariance and correlation matrices
^^^^

We can use SciDB's distributed parallel linear algebra operations to
compute covariance and correlation matrices without too much difficulty.
The following example computes the covariance and correlation between
the columns of the matrices X and Y. We break the example up into
a few parts for clarity.

Part 1, set up some example matrices::

  from scidbpy import interface, SciDBQueryError, SciDBArray
  sdb = interface.SciDBShimInterface('http://localhost:8080')
  import numpy as np

  # Create a small test array with 5 columns:
  X = sdb.from_array(np.random.random((10,5)))

  # Create a second small test array with 10 columns:
  Y = sdb.from_array(np.random.random((10,10)))

Part 2, center the example matrices::

  # Subtract the column means from X using broadcasting:
  xcolmean = X.mean(1)
  XC = X - xcolmean

  # Similarly subtract the column means from Y:
  ycolmean = Y.mean(1)
  YC = Y - ycolmean

Part 3, compute the covariance matrix::

  COV = sdb.dot(XC.T, YC)/(X.shape[0] - 1)

Part 4, compute the correlation matrix::

  # Column vector with column standard deviations of X matrix:
  xsd = X.std(1).reshape((5,1))
  # Row vector with column standard deviations of Y matrix:
  ysd = Y.std(1).reshape((1,10))
  # Their outer product:
  outersd = sdb.dot(xsd,ysd)

  COR = COV/outersd

The overhead of working interactively with SciDB makes these examples run
pretty slowly for tiny problems. But the same code shown here can be applied to
arbitrarily large matrices, and those computations can run in parallel across a
cluster.

