import sys

from . import SciDBArray
from .afldb import operators

mod = sys.modules[__name__]

__all__ = ['build_operator']

"""
Things to check:

* Do all AFL operators return arrays?
* Is it a good strategy to store all evaluations into new arrays,
  and cache them?
"""


def _format_operand(o):
    """
    Format the input into a string
    approproate for use as an input to an AFLExpression call

    Parameters
    ----------
    o : AFLExpression, SciDBArray, number, or string
        The input to format

    Returns
    -------
    result : str
        The formatted string
    """
    if isinstance(o, basestring):
        return o

    if isinstance(o, AFLExpression):
        if o.cached_result is not None:
            return o.cached_result.name
        return o.query

    if isinstance(o, SciDBArray):
        return o.name

    return str(o)


class AFLExpression(object):
    _signature = []

    def __init__(self, *args):
        self._check_arguments(args)
        self.args = args
        self._result = None

    def _check_arguments(self, args):
        ngiven = len(args)
        exact = self._signature[-1] != 'args'
        nrequired = len(self._signature) - 1 + int(exact)

        if ngiven < nrequired:
            arg = "argument" if nrequired == 1 else "arguments"
            constraint = "exactly" if exact else "at least"
            name = self.name
            raise TypeError("{name}() takes {constraint} {nrequired} {arg} "
                            "({ngiven} given)".format(**locals()))

    @property
    def name(self):
        """
        The SciDB operator name, assumed to be the same as the class name
        """
        return self.__class__.__name__

    @property
    def interface(self):
        # XXX This doesn't work for queries that don't use SciDBArrays
        for a in self.args:
            if hasattr(a, 'interface'):
                return a.interface
        raise ValueError("Could not find an interface")

    @property
    def query(self):
        ops = map(_format_operand, self.args)
        q = "{function}({args})".format(function=self.name, args=','.join(ops))
        return q

    @property
    def cached_result(self):
        """
        Return the result if already evaluated, or None.

        Returns
        -------
        A SciDBArray instance, or None
        """
        return self._result

    def eval(self, out=None):
        """
        Evaluate the expression if necessary, and return the
        result as a SciDBArray.

        Parameters
        ----------
        out : SciDBArray instance (optional)
            The array to store the result in.
            One will be created if necessary

        Returns
        --------
        out : SciDBArray instance

        Note
        ----
        The result of eval() is cached, so subsequent calls
        will not trigger additional database computation
        """
        if self._result is not None:
            return self._result

        if out is None:
            out = self.interface.new_array()
        self._result = out

        self.interface._execute_query('store(%s, %s)' %
                                      (self.query, out.name))
        return self._result

    def toarray(self):
        return self.eval().toarray()

    def __str__(self):
        return self.query

    def __repr__(self):
        return "SciDB Expression: <%s>" % self


def build_operator(entry):
    """
    Create a new AFL operator, based on a description dictionary

    Parameters
    ----------
    entry: dict with the following keys:
        - name : string giving the name of the operator
        - doc : docstring
        - signature : list giving the type of each argument

    Returns
    --------
    A new AFLExpression subclass, represenging the operator
    """
    name = str(entry['name'])
    doc = entry['doc']
    signature = entry['signature']

    attrs = {'__doc__': doc, '_signature': signature}
    result = type(name, (AFLExpression,), attrs)
    return result


# build the functions and populate the namespace
for op in operators:
    setattr(mod, op['name'], build_operator(op))
    __all__.append(op['name'])
