import pytest

import schema


class TestAttribute:

    @pytest.mark.parametrize(
        ('string', 'expected'),
        [
            ('foo:int64', schema.Attribute('foo', 'int64')),
            ('foo:int64 not null',
             schema.Attribute('foo', 'int64', not_null=True)),
            ('foo:int64 default 10',
             schema.Attribute('foo', 'int64', default='10')),
            ('foo:int64 compression zlib',
             schema.Attribute('foo', 'int64', compression='zlib')),
        ])
    def test_fromstring(self, string, expected):
        assert schema.Attribute.fromstring(string) == expected


class TestDimension:

    @pytest.mark.parametrize(
        ('string', 'expected'),
        [
            ('No=0:*:0:1000000', schema.Dimension('No', 0, '*', 0, 1000000)),
        ])
    def test_fromstring(self, string, expected):
        assert schema.Dimension.fromstring(string) == expected


class TestSchema:

    @pytest.mark.parametrize(
        ('string', 'expected'),
        [
            ('list<name:string NOT NULL,uaid:int64 NOT NULL,' +
             'aid:int64 NOT NULL,schema:string NOT NULL,' +
             'availability:bool NOT NULL,temporary:bool NOT NULL> ' +
             '[No=0:*:0:1000000]',
             schema.Schema(
                 'list',
                 (schema.Attribute('name', 'string', True),
                  schema.Attribute('uaid', 'int64', True),
                  schema.Attribute('aid', 'int64', True),
                  schema.Attribute('schema', 'string', True),
                  schema.Attribute('availability', 'bool', True),
                  schema.Attribute('temporary', 'bool', True)),
                 (schema.Dimension('No', 0, '*', 0, 1000000),)))
        ])
    def test_fromstring(self, string, expected):
        assert schema.Schema.fromstring(string) == expected
