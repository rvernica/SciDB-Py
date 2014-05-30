AFL Operator Reference
======================
The ``afl`` namespace provides direct access to SciDB
AFL operators. More information on these operators can be found
on the `SciDB documentation <http://scidb.org/HTMLmanual/14.3/scidb_ug/ch17.html>`_.

AFL Operators can be used as follows::

    sdb = connect()
    x = sdb.random((3, 4))
    afl = sdb.afl
    query = afl.aggregate(x, 'count(*)')
    query.query  # The query string
    query.eval()  # evaluate the query, and store the result


Operator Reference
------------------
.. automodule:: scidbpy.afl
   :members:
