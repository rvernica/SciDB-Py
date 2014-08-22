.. currentmodule:: scidbpy

.. _whatsnew:

=========
Whats New
=========

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