.. currentmodule:: scidbpy

.. _whatsnew:

=========
Whats New
=========

16.9.dev0 (Release September 9, 2017)
--------------------------------

Fix SciDB ``16.9`` compatibility.

15.12 (Released March 21, 2016)
---------------------------------

Highlights
^^^^^^^^^^
- now supports SciDB15.12; use branch `scidb15.7` for use with SciDB15.7

14.10 (Released November 4, 2014)
---------------------------------

Highlights
^^^^^^^^^^
- Support for performing :meth:`~SciDBArray.groupby` on non-integer attributes
- Added a pandas-like :meth:`~interface.SciDBInterface.merge` method
  to perform database-style joins
- Experimental support for :ref:`compressed transfer <compressed_transfer>`,
  and for :ref:`efficient downloading <dense_transfer>` of dense arrays
- Added :ref:`robust versions <robust>` of many AFL operators, that perform array
  preprocessing (chunk alignment, schema matching, etc.) automatically
- Several new methods: :meth:`~interface.SciDBInterface.unique`, :meth:`~interface.SciDBInterface.percentile`, :meth:`~SciDBArray.isel`, :meth:`~interface.SciDBInterface.hstack`, :meth:`~interface.SciDBInterface.vstack`, :meth:`~interface.SciDBInterface.concatenate`, :meth:`~SciDBArray.any`, :meth:`~SciDBArray.all`, :meth:`~interface.SciDBInterface.remove`, :meth:`~interface.SciDBInterface.ls`, :meth:`~SciDBArray.collapse`
- Ability to index arrays using :ref:`integer arrays <fancy_indexing>`

API Changes in 14.10
^^^^^^^^^^^^^^^^^^^^
 - Slicing an array by a boolean array now produces a sparse result, preserving
   the location of the selected cells. The collapse method converts this array
   to the 1D dense array previously returned by boolean masking.
 - The :meth:`~SciDBPy.interface.SciDBInterface.merge` method was changed
   from a direct AFL call to a high-level join operator. Use robust.merge
   for the old behavior

14.8 (Released August 22, 2014)
-------------------------------

Highlights
^^^^^^^^^^

- Support for :ref:`authenticated and encrypted <authentication>`
  connections to ScIDB

- Fixed a bug where uploading large arrays using `from_data` resulted
  in scrambled cell locations in the database

- Proper treatment of elementwise arithmetic on sparse arrays

14.7 (Released August 1, 2014)
------------------------------

Highlights
^^^^^^^^^^

 - Wider support for :ref:`wrapping <downloading>` all of SciDB's built-in
   datatypes, including strings, datetimes, and nullable values::

       >>> x
       SciDBArray('py1101328071989_00045<f0:string> [i0=0:2,1000,0]')
       >>> x.toarray()
       array([u'abc', u'def', u'ghi'], dtype=object)


 - A :meth:`~SciDBArray.groupby` method for performing aggregation over groups::

     x.groupby('gender').aggregate('mean(age)')

 - :ref:`Boolean comparison and filtering of arrays <comparison_and_filtering>`::

     >>> x = sdb.random((5,5))
     >>> (x > 0.7).toarray()
     array([[ True,  True, False, False, False],
            [ True,  True,  True, False, False],
            [False, False, False,  True,  True],
            [False, False,  True, False, False],
            [False,  True,  True, False, False]], dtype=bool)
     >>> x[x>0.7].toarray()
     array([ 0.83500619,  0.95791602,  0.94745933,  0.89868099,  0.97664716,
             0.7045693 ,  0.88949448,  0.88112397,  0.73766701,  0.94612052])

 - Pandas-like :ref:`syntax<attribute_access>` for accessing and defining new
   array attributes::

       array['b'] = 'sin(f0+3)'
       array['b'].toarray()

 - :ref:`Lazy evaluation<lazy>` of arrays. Computation for most array operations
   are deferred until needed.

 - AFL queries now return lazy SciDBArrays instead of special class instances,
   which makes it easy to mix SciDBArray methods with raw AFL calls::

       >>> sdb.afl.build('<x:float>[i=0:5,10,0]', 'i').max()[0]

 - A cleaner syntax for :ref:`chaining <afl_chain>` several AFL calls at once.
   The following two lines are equivalent::

       f = sdb.afl
       f.subarray(f.project(f.apply(x, 'f2', 'f*2'), 'f2'), 0, 5)

       x.apply('f2', 'f*2').project('f2').subarray(0, 5)

 - New element-wise operators: sqrt, floor, ceil, isnan

 - A :meth:`~SciDBArray.cumulate` method for performing cumulative
   aggregation over arrays

 - Numerous bugfixes.
