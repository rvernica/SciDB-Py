# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from __future__ import print_function, unicode_literals

import sys
import threading

from . import SciDBArray
from .afldb import operators
from ._py3k_compat import string_type

_mod = sys.modules[__name__]

__all__ = ['register_operator', 'AFLNamespace']


def _format_operand(o):
    """
    Format the input into a string
    appropriate for use as an input to an AFL call

    Parameters
    ----------
    o : SciDBArray, number, or string
        The input to format

    Returns
    -------
    result : str
        The formatted string
    """

    # awkward: commands like list need their strings to be single-quoted
    #          for now, this has to be done manually
    if isinstance(o, string_type):
        return o

    if isinstance(o, SciDBArray):
        return o.name

    return str(o)


def _query_string(operator, args):
    args = ','.join(map(_format_operand, args))
    return "{operator}({args})".format(operator=operator, args=args)


def _default_interface():
    return getattr(threading.local(), 'interface', None)


def _find_interface(args):
    for a in args:
        if hasattr(args, 'interface'):
            return args.interface


def afl_call(operator, interface, *args):
    interface = interface or _find_interface(args) or _default_interface()

    if interface is None:
        raise ValueError("No SciDBInterface provided, and cannot be inferred "
                         "from input arguments")

    query = _query_string(operator, args)
    return SciDBArray.from_query(interface, query)


def infix_call(operator, left, right):
    query = ' '.join([_format_operand(left), operator, _format_operand(right)])
    return query


def register_operator(entry, interface=None):
    """
    Create a new AFL operator, based on a description dictionary

    Parameters
    ----------
    entry: dict with the following keys:
        - name : string giving the name of the operator
        - doc : docstring
        - signature : list giving the type of each argument

    interface : SciDBinterface instance
        Which SciDBinterface instance to bind the operator class to

    Returns
    --------
    A new afl wrapper function, representing the operator
    """

    # we avoid partial here because its repr isn't great
    def call(*args):
        return afl_call(entry['name'], interface, *args)

    call.__doc__ = str(entry['doc'])
    call.__name__ = str(entry['name'])

    return call


def register_infix(name, op):
    """
    Create a new infix operator like '<', '+', etc.

    Parameters
    ----------
    name : str
        Name of the class (e.g., 'add')
    op : str
        SciDB operator name (e.g., '+')

    Returns
    -------
    A new function representing the operator
    """
    def call(*args):
        return infix_call(op, *args)
    call.__doc__ = str("The operator %s" % op)
    call.__name__ = str(name)

    return call


class AFLNamespace(object):

    """
    A module-like namespace for all registered operators.
    """

    def __init__(self, interface):
        for op in operators:
            if op['name'] in DEPRECATED:
                continue
            setattr(self, op['name'], register_operator(op, interface))

        for name, op in infix_functions:
            setattr(self, name, register_infix(name, op))

    def papply(self, array, attr, expression):
        """
        papply(array, attr, expression)

        Shorthand for project(apply(array, attr, expression), attr)
        """
        return self.project(self.apply(array, attr, expression), attr)

    def quote(self, val):
        """Wrap the argument in single quotes.

        Useful for AFL operators which expected quoted strings
        """
        return "'%s'" % val

    def count(self, array):
        # replace count, which was removedin SciDB 14
        return self.aggregate(array, 'count(*)')

    def redimension_store(self, arr_in, arr_out):
        # replace redimension_store, removed in SciDB 14
        if (not isinstance(arr_in, SciDBArray) or
                not isinstance(arr_out, SciDBArray)):
            raise TypeError("Inputs to redimension_store must be SciDB arrays")

        return self.store(self.redimension(arr_in, arr_out),
                          arr_out)


# tuple of (python AFL name, scidb token) for binary infix functions
infix_functions = [('as_', 'as'), ('add', '+'),
                   ('sub', '-'), ('mul', '*'), ('div', '/'),
                   ('mod', '%'), ('lt', '<'), ('le', '<='),
                   ('ne', '<>'), ('eq', '='), ('ge', '>='), ('gt', '>')]

# Functions that are no longer supported
DEPRECATED = set()

# scalar functions
# TODO grab docstrings from these somewhere?
functions = ['abs', 'acos', 'and', 'append_offset', 'apply_offset', 'asin',
             'atan', 'ceil', 'cos', 'day_of_week', 'exp', 'first_index',
             'floor', 'format', 'get_offset', 'high', 'hour_of_day',
             'iif', 'instanceid', 'is_nan', 'is_null', 'last_index', 'length',
             'log', 'log10', 'low', 'max', 'min', 'missing', 'missing_reason',
             'not', 'now', 'or', 'pow', 'random', 'regex', 'sin', 'sqrt',
             'strchar', 'strftime', 'strip_offset', 'strlen', 'substr',
             'tan', 'togmt', 'tznow']
for f in functions:
    operators.append(dict(name=f, signature=[],
                          doc='The scalar function %s' % f))

# add in some missing operators from other libraries
# TODO add these to afldb.py
for op in ['gemm', 'gesvd']:
    operators.append(dict(name=op, signature=[], doc=''))


# for documentation purposes, create operator classes
# unattached to references. These aren't generally useful, but
# this lets sphinx find and document each class
for op in operators:
    setattr(_mod, op['name'], register_operator(op))
    __all__.append(op['name'])

for name, op in infix_functions:
    setattr(_mod, name, register_infix(name, op))
    __all__.append(name)
