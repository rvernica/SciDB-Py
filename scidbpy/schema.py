import itertools
import logging
import numpy
import re
import struct


type_map = dict(
    [(t.__name__, t) for t in (
        numpy.bool,
        numpy.double,

        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,

        numpy.uint8,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
    )] + [
        ('char', 'a1'),
        ('float', numpy.float32),
        ('string', numpy.object),
    ])

# TODO datetime, datetimetz


class Attribute(object):
    """Represent SciDB array attribute

    Construct an attribute using Attribute constructor:

    >>> Attribute('foo', 'int64', not_null=True)
    Attribute('foo', 'int64', True, None, None)

    >>> Attribute('foo', 'int64', default=100, compression='zlib')
    Attribute('foo', 'int64', False, 100, 'zlib')


    Construct an attribute from a string:

    >>> Attribute.fromstring('foo:int64')
    Attribute('foo', 'int64', False, None, None)

    >>> Attribute.fromstring(
    ...     "taz : string NOT null DEFAULT '' compression bzlib")
    Attribute('taz', 'string', True, "''", 'bzlib')
    """

    _regex = re.compile('''
        \s*
        (?P<name>      \w+ ) \s* : \s*
        (?P<type_name> \w+ ) \s*
        (?:                 (?P<not_null>    NOT )? \s+ NULL )? \s*
        (?: DEFAULT     \s+ (?P<default>     \S+ )           )? \s*
        (?: COMPRESSION \s+ (?P<compression> \w+ )           )? \s*
        $''', re.VERBOSE | re.IGNORECASE)
    # length dtype for vairable-size ScidDB types
    _length_dtype = numpy.dtype(numpy.uint32)

    def __init__(self,
                 name,
                 type_name,
                 not_null=False,
                 default=None,
                 compression=None):
        self.name = name
        self.type_name = type_name
        self.not_null = bool(not_null)
        self.default = default
        self.compression = compression

        self.val_dtype = type_map.get(self.type_name, numpy.object)
        # >>> numpy.dtype([(u"a", int)])
        # TypeError: data type not understood
        # https://github.com/numpy/numpy/issues/2407
        if self.not_null:
            self.dtype = numpy.dtype([(str(self.name), self.val_dtype)])
        else:
            self.dtype = numpy.dtype([(str(self.name),
                                       [('null', numpy.uint8),
                                        ('val', self.val_dtype)])])
        self.fmt = '{}{}'.format(self.type_name,
                                 '' if self.not_null else ' null')

    def __iter__(self):
        return (i for i in (
            self.name,
            self.type_name,
            self.not_null,
            self.default,
            self.compression))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__, *self)

    def __str__(self):
        return '{}:{}{}{}{}'.format(
            self.name,
            self.type_name,
            ' NOT NULL' if self.not_null else '',
            ' DEFAULT {}'.format(self.default) if self.default else '',
            ' COMPRESSION {}'.format(self.compression)
            if self.compression else '')

    def itemsize(self, buf=None, offset=0):
        if self.val_dtype != numpy.object:
            return self.dtype.itemsize

        null_size = 0 if self.not_null else 1
        value_size = numpy.frombuffer(
            buf, numpy.uint32, 1, offset + null_size)[0]
        return null_size + Attribute._length_dtype.itemsize + value_size

    def frombytes(self, buf, offset=0, size=None):
        null_size = 0 if self.not_null else 1

        if self.val_dtype == numpy.object:
            if self.type_name == 'string':
                val = buf[offset + null_size +
                          Attribute._length_dtype.itemsize:
                          offset + size - 1].decode()
            else:
                val = buf[offset + null_size +
                          Attribute._length_dtype.itemsize:
                          offset + size]
        else:
            val = numpy.frombuffer(
                buf, self.val_dtype, 1, offset + null_size)[0]

        if self.not_null:
            return val
        else:
            return (struct.unpack('B', buf[offset:offset + null_size])[0], val)

    @classmethod
    def fromstring(cls, string):
        return cls(**Attribute._regex.match(string).groupdict())


class Dimension(object):
    """Represent SciDB array dimension

    Construct a dimension using the Dimension constructor:

    >>> Dimension('foo')
    Dimension('foo', None, None, None, None)

    >>> Dimension('foo', -100, '10', '?', '1000')
    Dimension('foo', -100, 10, '?', 1000)


    Construct a dimension from a string:

    >>> Dimension.fromstring('foo')
    Dimension('foo', None, None, None, None)

    >>> Dimension.fromstring('foo=-100:*:?:10')
    Dimension('foo', -100, '*', '?', 10)
    """

    _regex = re.compile('''
        \s*
        (?P<name> \w+ ) \s*
        (?: = \s* (?P<low_value>  [^:\s]+ ) \s*
                  : \s*
                  (?P<high_value> [^:\s]+ ) \s*
                  (?: : \s* (?P<chunk_overlap> [^:\s]+ ) \s*
                            (?: : \s* (?P<chunk_length> [^:\s]+ ) )?
                  )?
        )?
        \s* $''', re.VERBOSE)

    def __init__(self,
                 name,
                 low_value=None,
                 high_value=None,
                 chunk_overlap=None,
                 chunk_length=None):
        self.name = name

        try:
            self.low_value = int(low_value)
        except (TypeError, ValueError):
            self.low_value = low_value

        try:
            self.high_value = int(high_value)
        except (TypeError, ValueError):
            self.high_value = high_value

        try:
            self.chunk_overlap = int(chunk_overlap)
        except (TypeError, ValueError):
            self.chunk_overlap = chunk_overlap

        try:
            self.chunk_length = int(chunk_length)
        except (TypeError, ValueError):
            self.chunk_length = chunk_length

    def __iter__(self):
        return (i for i in (
            self.name,
            self.low_value,
            self.high_value,
            self.chunk_overlap,
            self.chunk_length))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__, *self)

    def __str__(self):
        out = self.name
        if self.low_value is not None:
            out += '={}:{}'.format(self.low_value, self.high_value)
            if self.chunk_overlap is not None:
                out += ':{}'.format(self.chunk_overlap)
                if self.chunk_length is not None:
                    out += ':{}'.format(self.chunk_length)
        return out

    @classmethod
    def fromstring(cls, string):
        return cls(**Dimension._regex.match(string).groupdict())


class Schema(object):
    """Represent SciDB array schema

    Cunstruct a schema using Schema, Attribute, and Dimension
    constructors:

    >>> Schema('foo', (Attribute('x', 'int64'),), (Dimension('i', 0, 10),))
    ... # doctest: +NORMALIZE_WHITESPACE
    Schema('foo',
           (Attribute('x', 'int64', False, None, None),),
           (Dimension('i', 0, 10, None, None),))


    Construct a schema using Schema constructor and fromstring methods
    of Attribute and Dimension:

    >>> Schema('foo',
    ...        (Attribute.fromstring('x:int64'),),
    ...        (Dimension.fromstring('i=0:10'),))
    ... # doctest: +NORMALIZE_WHITESPACE
    Schema('foo',
           (Attribute('x', 'int64', False, None, None),),
           (Dimension('i', 0, 10, None, None),))


    Construct a schema from a string:

    >>> Schema.fromstring(
    ...     'foo@1<x:int64 not null, y:double>[i=0:*; j=-100:0:0:10]')
    ... # doctest: +NORMALIZE_WHITESPACE
    Schema('foo@1',
           (Attribute('x',  'int64',  True, None, None),
            Attribute('y', 'double', False, None, None)),
           (Dimension('i',    0, '*', None, None),
            Dimension('j', -100,   0,    0, 10)))


    Print a schema constructed from a string:

    >>> print(Schema.fromstring('<x:int64,y:float> [i=0:2:0:1000000; j=0:*]'))
    ... # doctest: +NORMALIZE_WHITESPACE
    <x:int64,y:float> [i=0:2:0:1000000; j=0:*]


    Format Schema object to only print the schema part without the
    array name:

    >>> '{:h}'.format(Schema.fromstring('foo<x:int64>[i]'))
    '<x:int64> [i]'
    """

    _regex_name = re.compile('\s* (?P<name> [^<]+ )?', re.VERBOSE)

    _regex_atts = re.compile(
        '\s*  < ( [^,>]+  \s* (?: , \s* [^,>]+  \s* )* )  >', re.VERBOSE)

    _regex_dims = re.compile(
        '\s* \[ ( [^;\]]+ \s* (?: ; \s* [^;\]]+ \s* )* ) \] \s* $',
        re.VERBOSE)

    def __init__(self, name=None, atts=(), dims=()):
        self.name = name
        self.atts = tuple(atts)
        self.dims = tuple(dims)
        self._update_atts_meta()

    def __iter__(self):
        return (i for i in (self.name, ) + self.atts + self.dims)

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __repr__(self):
        return '{}({!r}, {!r}, {!r})'.format(
            type(self).__name__, self.name, self.atts, self.dims)

    def __str__(self):
        return self._render()

    def __format__(self, fmt_spec=''):
        return self._render(no_name='h' in fmt_spec)

    def _render(self, no_name=False):
        return '{}<{}> [{}]'.format(
            self.name if not no_name and self.name else '',
            ','.join(str(a) for a in self.atts),
            '; '.join(str(d) for d in self.dims))

    def make_dims_unique(self):
        """Check attributes and dimensions form a list with unique names. If
        not, rename diemsnions. Return True is any dimension was
        renamed.

        >>> s = Schema(None, (Attribute('i', 'bool'),), (Dimension('i'),))
        >>> print(s)
        <i:bool> [i]
        >>> s.make_dims_unique()
        True
        >>> print(s)
        <i:bool> [i_1]

        >>> s = Schema.fromstring('<i:bool>[i;i_1]')
        >>> s.make_dims_unique()
        True
        >>> print(s)
        <i:bool> [i_2; i_1]
        """
        all_names = set(itertools.chain((a.name for a in self.atts),
                                        (d.name for d in self.dims)))
        if len(all_names) < len(self.atts) + len(self.dims):
            # Rename dimensions
            atts_names = set(a.name for a in self.atts)
            new_dims_names = []
            for d in self.dims:
                # Check if dimension name collides with any attribute
                # name
                if d.name in atts_names:
                    count = 1
                    new_dim_name = '{}_{}'.format(d.name, count)
                    while new_dim_name in all_names:
                        count += 1
                        new_dim_name = '{}_{}'.format(d.name, count)
                    d.name = new_dim_name
            return True
        else:
            return False

    def make_dims_attr(self):
        """Make attributes from dimensions and append them to the attributes
        list.

        >>> s = Schema(None, (Attribute('x', 'bool'),), (Dimension('i'),))
        >>> print(s)
        <x:bool> [i]
        >>> s.make_dims_attr()
        >>> print(s)
        <x:bool,i:int64 NOT NULL> [i]

        >>> s = Schema.fromstring('<x:bool>[i;j]')
        >>> s.make_dims_attr()
        >>> print(s)
        <x:bool,i:int64 NOT NULL,j:int64 NOT NULL> [i; j]
        """
        self.atts = tuple(itertools.chain(
            self.atts,
            (Attribute(d.name, 'int64', not_null=True) for d in self.dims)))
        self._update_atts_meta()

    def _update_atts_meta(self):
        self.atts_dtype = numpy.dtype(
            list(
                itertools.chain.from_iterable(
                    a.dtype.descr for a in self.atts)))
        self.atts_fmt = '({})'.format(', '.join(a.fmt for a in self.atts))

    @classmethod
    def fromstring(cls, string):
        name_match = Schema._regex_name.match(string)
        atts_match = Schema._regex_atts.match(string, name_match.end(0))
        dims_match = Schema._regex_dims.match(string, atts_match.end(0))

        name = name_match.groupdict()['name']
        return cls(
            name.strip() if name else None,
            (Attribute.fromstring(s)
             for s in atts_match.group(1).split(',')),
            (Dimension.fromstring(s)
             for s in dims_match.group(1).split(';')))


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE)
