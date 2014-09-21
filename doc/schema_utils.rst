
.. _schema_utils:

Schema Manipulation Utilities
=============================

.. currentmodule:: scidbpy

The :mod:`scidbpy.schema_utils` module contains functions useful for
manipulating array schemas. Many native SciDB functions require that
the schemas of input arrays obey certain properties, like having
identical chunk sizes. The routines in this module help to preprocess
arrays to satisfy these requirements.

The functions in this module are designed to return their inputs unchanged,
if no modification is necessary. This saves you from having to pre-check
whether a given preprocessing step is necessary.

See :ref:`robust` for a collection of SciDB-Py analogs to AFL functions,
which perform necessary array preprocessing automatically.


Functions
---------

.. automodule:: scidbpy.schema_utils
   :members: