.. _downloading:


Downloading SciDBArrays
========================

.. currentmodule:: scidbpy

It is often advantageous to convert small arrays into
"normal" python data structures, for further analysis in python.
SciDB-Py provides several conversion routines:

 * :meth:`SciDBArray.toarray` converts the array to a NumPy array
 * :meth:`SciDBArray.todataframe` converts the array to a Pandas DataFrame
 * :meth:`SciDBArray.tosparse` converts the array to a SciPy sparse array

SciDB supports a wide variety of data types and array schemas, including
several concepts that don't have obvious analogs in NumPy. These include:

 * Nonzero array origins
 * Unbounded array dimensions
 * Null values for all datatypes (including integers)

The exact shape and datatype of the result of
:meth:`SciDBArray.toarray` depends on these details. Below
we outline the various possibilities, starting with the easiest cases.


Single attribute, non-sparse, non-nullable array
-------------------------------------------------
An array with a single non-nullable attribute is converted
into a NumPy array of equivalent datatype:

    ============== =========================
    SciDB datatype NumPy datatype (typecode)
    ============== =========================
    bool           bool ('<b1')
    int8           int8 ('<b')
    uint8          uint8 ('<B')
    int16          int16 ('<h')
    uint16         uint16 ('<H')
    int32          int32 ('<i')
    uint32         uint32 ('<I')
    int64          int64 ('<l')
    uint64         uint64 ('<L')
    float          float32 ('<float32')
    double         double ('<d')
    char           S1 ('c')
    datetime       datetime64 ('<M8[s]')
    datetimetz     datetime64 ('<M8[s]')
    string         object
    ============== =========================

Note that strings are converted into python object arrays, since
NumPy string arrrays are otherwise required to have the same string
length in each element

Single attribute, nullable array
---------------------------------
NumPy does not support the notion of missing values for datatypes
like integers and boolean. Thus, when downloading an array with
a nullable attribute, these datatypes are "promoted" to a datatype
with a dedicated missing value holder:

============== =========================
SciDB datatype Null-promoted datatype
============== =========================
bool           double
int8           double
uint8          double
int16          double
uint16         double
int32          double
uint32         double
int64          double
uint64         double
float          float32
double         double
char           S1 ('c')
datetime       datetime64 ('<M8[s]')
datetimetz     datetime64 ('<M8[s]')
string         object
============== =========================

In the NumPy array, each masked element is assigned the default null
value for its datatype:

==============  ====================
NumPy Datatype  Default masked value
==============  ====================
float, double   NaN
char            ``'\0'``
object          None
datetime        NaT
==============  ====================

Another way to deal with missing values is to substitute
a manually-defined missing value. This converts the array to a
non-nullable array::

    >>> from scidbpy import connect
    >>> sdb = connect()

    >>> x = sdb.afl.build('<a:int8 NULL>[i=0:5,10,0]', 'iif(i>0, i, null)')
    >>> x.toarray()
    array([ nan,   1.,   2.,   3.,   4.,   5.])

    >>> x.substitute(-1).toarray()
    array([-1,  1,  2,  3,  4,  5], dtype=int8)

SciDB allows several different "missing-data" codes to be assigned
to a masked cell. At the moment SciDB-Py doesn't distinguish between these:
either a cell has data, or it is considered masked.

Arrays with empty cells
-----------------------
In addition to masked values, SciDBArrays can have empty cells.
These cells are treated as zero-valued when converting to a NumPy array.
The zero-value for non-numeric datatypes is determined from the NumPy.zeros function::

    >>> x = sdb.afl.build('<a:int8>[i=0:3,10,0]', 10)
    >>> x = x.redimension('<a:int8>[i=0:5,10,0]')
    >>> x.toarray()
    array([10, 10, 10, 10,  0,  0], dtype=int8)

    >>> x = sdb.afl.build('<a:char>[i=0:3,10,0]', "'a'")
    >>> x = x.redimension('<a:char>[i=0:5,10,0]')
    >>> x.toarray()
    array(['a', 'a', 'a', 'a', '', ''], dtype='|S1')

Nonzero Origins
---------------
:meth:`SciDBArray.toarray` shifts any non-zero origin to the 0-position
in the NumPy array::

    >>> x = sdb.afl.build('<a:int8>[i=5:10,10,0]', 'i')
    >>> x.toarray()
    array([ 5,  6,  7,  8,  9, 10], dtype=int8)

The original array indices can be extracted using the unpack operator::

    >>> x.unpack('_idx').toarray()
    array([(5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)],
           dtype=[('i', '<i8'), ('a', 'i1')])

Unbound Arrays
--------------
When a SciDBArray is unbound, the resulting NumPy array is truncated
to the region containing data.

Multiattribute arrays
---------------------
Arrays with multiple attributes are handled analogously to single-attribute
arrays as discussed above. However, the output is returned as a NumPy
record array, with record labels matching the SciDB attribute labels.
