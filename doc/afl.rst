
.. _afl:

AFL Operator Reference
======================

.. currentmodule:: scidbpy

This is a list of functions in SciDB-Py's AFL binding. They are all
accessible under the ``afl`` attribute from the interface object returned by ``connect``, eg::

    from scidbpy import connect
    sdb = connect()
    afl = sdb.afl
    afl.adddim(...)

See :ref:`using_afl` for more information.

List of Operators
^^^^^^^^^^^^^^^^^

.. include:: afldb.rst
