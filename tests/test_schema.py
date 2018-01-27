import pytest

from scidbpy.schema import Attribute, Dimension, Schema


class TestAttribute:

    @pytest.mark.parametrize(
        ('args', 'expected'),
        [
            ({'name': 'bar',
              'type_name': 'float',
              'default': 3.14,
              'compression': 'zlib'},
             Attribute('bar', 'float', False, 3.14, 'zlib')),
            ({'name': 'foo',
              'type_name': 'int64',
              'default': 100},
             Attribute('foo', 'int64', False, 100, None)),
        ])
    def test_constructor(self, args, expected):
        assert Attribute(**args) == expected

    @pytest.mark.parametrize(
        ('string', 'expected'),
        (('{}:{}{}{}{}'.format(
            name,
            type_name,
            ' not null' if not_null else ' NULL',
            ' DEFAULT {}'.format(default) if default is not None else '',
            (" compression '{}'".format(compression)
             if compression is not None else '')),
          Attribute(name, type_name, not_null, default, compression))
         for name in ('x', 'foo', 'BAR')
         for type_name in ('int64', 'float', 'double')
         for not_null in (True, False)
         for default in (None, '10', '3.14', "''")
         for compression in (None, 'zlib', 'bzlib')))
    def test_fromstring(self, string, expected):
        assert Attribute.fromstring(string) == expected

    @pytest.mark.parametrize(
        ('string', 'expected'),
        [
            ('foo : int64',
             {'name': 'foo',
              'type_name': 'int64',
              'not_null': None,
              'default': None,
              'compression': None}),
            ('i : float not null',
             {'name': 'i',
              'type_name': 'float',
              'not_null': 'not',
              'default': None,
              'compression': None}),
            ("foo : string NULL DEFAULT '' COMPRESSION 'zlib'",
             {'name': 'foo',
              'type_name': 'string',
              'not_null': None,
              'default': "''",
              'compression': 'zlib'}),
        ])
    def test_regex(self, string, expected):
        assert Attribute._regex.search(string).groupdict() == expected


class TestDimension:

    @pytest.mark.parametrize(
        ('args', 'expected'),
        [
            (('foo', 0, '*'),
             Dimension('foo', 0, '*', None, None)),
            (('foo', '?', '10'),
             Dimension('foo', '?', 10, None, None)),
            (('foo', '?', '10', '?'),
             Dimension('foo', '?', 10, '?', None)),
        ])
    def test_constructor(self, args, expected):
        assert Dimension(*args) == expected

    @pytest.mark.parametrize(
        ('string', 'expected'),
        [('{}{}{}{}{}'.format(
            name,
            '={}'.format(low_value) if low_value is not None else '',
            ':{}'.format(high_value) if high_value is not None else '',
            ':{}'.format(chunk_overlap) if chunk_overlap is not None else '',
            ':{}'.format(chunk_length) if chunk_length is not None else ''),
          Dimension(
              name, low_value, high_value, chunk_overlap, chunk_length))

         for name in ('i', 'foo', 'BAR')
         for low_value in (None, '?', -100, 0, 10)
         for high_value in (None, '?', '*', -100, 0, 10)
         for chunk_overlap in (None, '?', 0, 10)
         for chunk_length in (None, '?', 0, 10)
         if (low_value is not None and
             high_value is not None and
             (chunk_overlap is not None or
              chunk_overlap is None and chunk_length is None) or
             low_value is None and
             high_value is None and
             chunk_overlap is None and
             chunk_length is None)] +
        [('No=0:*:0:1000000', Dimension('No', 0, '*', 0, 1000000))])
    def test_fromstring(self, string, expected):
        assert Dimension.fromstring(string) == expected

    @pytest.mark.parametrize(
        ('string', 'expected'),
        [
            ('foo',
             {'name': 'foo',
              'low_value': None,
              'high_value': None,
              'chunk_overlap': None,
              'chunk_length': None}),
            ('foo = 1 : 22',
             {'name': 'foo',
              'low_value': '1',
              'high_value': '22',
              'chunk_overlap': None,
              'chunk_length': None}),
            ('foo = 1 : 22 : 333',
             {'name': 'foo',
              'low_value': '1',
              'high_value': '22',
              'chunk_overlap': '333',
              'chunk_length': None}),
            ('foo = 1 : 22 : 333 : 4444',
             {'name': 'foo',
              'low_value': '1',
              'high_value': '22',
              'chunk_overlap': '333',
              'chunk_length': '4444'}),
        ])
    def test_regex(self, string, expected):
        assert Dimension._regex.search(string).groupdict() == expected


class TestSchema:

    @pytest.mark.parametrize(
        ('string', 'expected_str', 'expected_obj'),
        [
            ('foo<x:int64>[i=0:*]',
             'foo<x:int64> [i=0:*]',
             Schema(
                 'foo',
                 (Attribute('x', 'int64'),),
                 (Dimension('i', 0, '*'),))),
            ('foo@10<x:int64>[i=0:*]',
             'foo@10<x:int64> [i=0:*]',
             Schema(
                 'foo@10',
                 (Attribute('x', 'int64'),),
                 (Dimension('i', 0, '*'),))),
            ('list<name:string NOT NULL,uaid:int64 NOT NULL,' +
             'aid:int64 NOT NULL,schema:string NOT NULL,' +
             'availability:bool NOT NULL,temporary:bool NOT NULL> ' +
             '[No=0:*:0:1000000]',
             'list<name:string NOT NULL,uaid:int64 NOT NULL,' +
             'aid:int64 NOT NULL,schema:string NOT NULL,' +
             'availability:bool NOT NULL,temporary:bool NOT NULL> ' +
             '[No=0:*:0:1000000]',
             Schema(
                 'list',
                 (Attribute('name', 'string', True),
                  Attribute('uaid', 'int64', True),
                  Attribute('aid', 'int64', True),
                  Attribute('schema', 'string', True),
                  Attribute('availability', 'bool', True),
                  Attribute('temporary', 'bool', True)),
                 (Dimension('No', 0, '*', 0, 1000000),))),
            ('not empty operators' +
             '<name:string NOT NULL,library:string NOT NULL> [No=0:52:0:53]',
             'operators' +
             '<name:string NOT NULL,library:string NOT NULL> [No=0:52:0:53]',
             Schema(
                 'operators',
                 (Attribute('name', 'string', True, None, None),
                  Attribute('library', 'string', True, None, None)),
                 (Dimension('No', 0, 52, 0, 53),))),
        ])
    def test_fromstring(self, string, expected_str, expected_obj):
        assert str(Schema.fromstring(string)) == expected_str
        assert Schema.fromstring(string) == expected_obj

    @pytest.mark.parametrize(
        ('string', 'expected'),
        [
            ('<x:int64,y:double>',
             ['x:int64', 'y:double']),
        ])
    def test_regex_atts(self, string, expected):
        assert (Schema._regex_atts.search(string).group(1).split(',') ==
                expected)

    @pytest.mark.parametrize(
        ('string', 'expected'),
        [
            ('[i=0:*;j=-100:0:0:10]',
             ['i=0:*', 'j=-100:0:0:10']),
        ])
    def test_regex_dims(self, string, expected):
        assert (Schema._regex_dims.search(string).group(1).split(';') ==
                expected)
