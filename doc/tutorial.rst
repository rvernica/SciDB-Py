.. _tutorial:

.. currentmodule:: scidbpy

=========
Basic Use
=========

The primary data structure in SciDB-Py is the :class:`SciDBArray`.
This object defines a NumPy array-like interface for SciDB arrays in
Python. While :class:`SciDBArray` syntax mimics NumPy, array
operations are converted to SciDB queries which are executed by the
database. Data are materialized to Python only when requested. A basic
set of array subsetting, arithmetic and utility operations are defined
by the package. Additionally, SciDB-Py provides several utilities for
compising SciDB queries more explicitly.

.. toctree::
   :maxdepth: 1

   connection
   creation
   access
   operations
   aggregation
   comparison_and_filtering
   query
   conversion.rst


