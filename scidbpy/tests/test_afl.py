# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from __future__ import absolute_import, print_function, division, unicode_literals

import pytest
import numpy as np

from scidbpy.afl import _format_operand
from . import sdb, TestBase, teardown_function

afl = sdb.afl


# create_array doxygen is malformed in SciDB 14.
@pytest.mark.xfail(reason="SciDB14 has a malformed docstring")
def test_docstring():
    """Regression test against unintentional docstring changes"""
    expected = "create_array( arrayName, arraySchema )\n\nCreates an array with a given name and schema.\n\nParameters\n----------\n\n    - arrayName: the array name\n    - arraySchema: the array schema of attrs and dims"
    assert afl.create_array.__doc__ == expected


def test_name():
    x = sdb.zeros((2, 2))
    expression = afl.transpose(afl.transpose(x))
    assert expression.name == "transpose(transpose(%s))" % x.name


class TestBasicUse(TestBase):

    def test_query(self):
        ARR = sdb.ones((4))
        assert afl.transpose(ARR).query == "transpose(%s)" % ARR.name

    def test_result(self):
        ARR = sdb.random((4))
        s = afl.sort(ARR)
        expected = np.sort(ARR.toarray()) 
        np.testing.assert_array_almost_equal(s.toarray(), expected)

    def test_results_cached(self):
        ARR = sdb.ones((4))
        s = afl.transpose(ARR)
        s.eval() is s.eval()

    def test_eval_nostore(self):
        ARR = sdb.ones((4))
        s = afl.transpose(ARR)
        s.eval(store=False)
        assert s.name not in sdb.list_arrays()

    def test_eval_output(self):
        ARR = sdb.random((4))
        s = afl.sort(ARR)
        out = sdb.new_array()
        result = s.eval(out=out)
        assert result.name is out.name
        expected = np.sort(ARR.toarray()) 
        np.testing.assert_array_almost_equal(s.toarray(), expected)

    def test_access_from_interface(self):
        assert sdb.afl.list("'arrays'").interface is sdb

    def test_interface_access_cached(self):
        assert sdb.afl is sdb.afl
        assert sdb.afl.transpose is sdb.afl.transpose

    def test_operator_chaining_syntax(self):
        a = afl
        x = sdb.random(10)
        y = x.apply('y', 'f0+1').project('y').filter('y > 3')
        z = a.filter(a.project(a.apply(x, 'y', 'f0+1'), 'y'), 'y > 3')
        assert y.query == z.query


class TestFormatOperands(TestBase):

    def test_scalar(self):
        assert _format_operand('x') == 'x'
        assert _format_operand(3) == '3'

    def test_scidbarray_givnen_by_name(self):
        ARR = sdb.ones((4))
        assert _format_operand(ARR) == ARR.name

    def test_expression_given_by_query(self):
        ARR = sdb.ones((4))
        assert _format_operand(afl.transpose(ARR)) == "transpose(%s)" % ARR.name

    def test_expression_given_by_result_name_if_evaluated(self):
        ARR = sdb.ones((4))

        exp = afl.transpose(ARR)
        exp.eval()
        assert _format_operand(exp) == exp.eval().name
