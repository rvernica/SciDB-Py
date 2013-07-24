"""
Errors and Exceptions
=====================
These are custom errors and exceptions for SciDB queries.
"""
# License: Simplified BSD, 2013
# Author: Jake Vanderplas <jakevdp@cs.washington.edu>

from collections import defaultdict


class SciDBError(Exception):
    pass


class SciDBInvalidQuery(SciDBError):
    pass


class SciDBInvalidSession(SciDBError):
    pass


class SciDBEndOfFile(SciDBError):
    pass


class SciDBInvalidRequest(SciDBError):
    pass


class SciDBQueryError(SciDBError):
    pass


class SciDBConnectionError(SciDBError):
    pass


class SciDBMemoryError(SciDBError):
    pass


class SciDBUnknownError(SciDBError):
    pass


SHIM_ERROR_DICT = defaultdict(lambda: SciDBUnknownError)
SHIM_ERROR_DICT[400] = SciDBInvalidQuery
SHIM_ERROR_DICT[404] = SciDBInvalidSession
SHIM_ERROR_DICT[410] = SciDBEndOfFile
SHIM_ERROR_DICT[414] = SciDBInvalidRequest
SHIM_ERROR_DICT[500] = SciDBQueryError
SHIM_ERROR_DICT[503] = SciDBConnectionError
SHIM_ERROR_DICT[507] = SciDBMemoryError
