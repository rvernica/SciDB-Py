.. currentmodule:: scidbpy

Introduction to SciDB arrays
----------------------------

SciDB arrays are composed of `cells`. Each cell may contain one or more values
referred to as `attributes`. The data types and number of attributes are
consistent across all cells within one array. All the attribute values within a
cell may be left undefined, in which case the cell is called empty. Arrays with
empty cells are referred to as sparse arrays in the SciDB documentation.

Individual attribute values may also be explicitly marked missing with one of
several possible SciDB null codes.

Cells are arranged by an integer coordinate system into n-dimensional arrays.
SciDB uses signed 64-bit integers for coordinates. Each coordinate axis is
typically referred to as a dimension in the SciDB documentation. SciDB is
limited in theory to about 100 dimensions, but in practice that limit is
typically much lower (up to say, 10 dimensions or so). While the default SciDB
array origin is usually zero, SciDB arrays may use any signed 64-bit
integer origin.

:class:`SciDBArray` objects are Python representations of
SciDB arrays that mimic numpy arrays in many ways.  :class:`SciDBArray` array
objects are limited to the following SciDB array attribute data types:
``bool``, ``float32``, ``float64``, ``int8``, ``int16``, ``int32``, ``int64``,
``uint8``, ``unit16``, ``uint32``, ``uint64``, characters, and strings.


Loading the scidbpy package and connecting to SciDB
---------------------------------------------------

In order to use SciDB, Python needs an interface to a SciDB
server.  This is accomlished through the :func:`~interface.connect` function.

connect takes an optional URL specifying the location of the SciDB
coordinator node (running Shim_; see :ref:`installing_scidbpy`). If no URL is provided, it looks for a ``SCIDB_URL`` environment variable, and then
defaults to ``http://localhost:8080``.

The following snippet imports SciDB-Py and establishes a connection
with the database -- adjust the
host name as required if SciDB is on a different computer::

   >>> import numpy as np
   >>> from scidbpy import connect
   >>> sdb = connect('http://localhost:8080')

Throughout this documentation, the ``sdb`` variable is used to refer
to the connection to a SciDB instance, as above.

.. _Shim: http://github.com/paradigm4/shim


