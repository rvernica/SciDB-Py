# License: Simplified BSD, 2014
# See LICENSE.txt for more information

import pytest

from mock import MagicMock
import numpy as np

from scidbpy import connect
from scidbpy.afl import _format_operand

sdb = connect()
afl = sdb.afl
ARR = sdb.ones((4))

# create_array doxygen is malformed in SciDB 14.


@pytest.mark.xfail(reason="SciDB14 has a malformed docstring")
def test_docstring():
    """Regression test against unintentional docstring changes"""
    expected = "create_array( arrayName, arraySchema )\n\nCreates an array with a given name and schema.\n\nParameters\n----------\n\n    - arrayName: the array name\n    - arraySchema: the array schema of attrs and dims"
    assert afl.create_array.__doc__ == expected


def test_tostring():
    expression = afl.normalize(afl.transpose(3))
    assert str(expression) == "normalize(transpose(3))"


def test_repr():
    expression = afl.normalize(afl.transpose(3))
    assert repr(expression) == "SciDB Expression: <normalize(transpose(3))>"


class TestArgumentChecks(object):

    def test_0_args_given_2_expected(self):
        with pytest.raises(TypeError) as cm:
            afl.concat()
        assert cm.value.args[0] == ("concat() takes exactly 2 "
                                    "arguments (0 given)")

    def test_0_args_given_1_expected(self):
        with pytest.raises(TypeError) as cm:
            afl.dimensions()
        assert cm.value.args[0] == ("dimensions() takes exactly 1 "
                                    "argument (0 given)")

    def test_0_args_given_at_least_1_expected(self):
        with pytest.raises(TypeError) as cm:
            afl.analyze()
        assert cm.value.args[0] == ("analyze() takes at least 1 "
                                    "argument (0 given)")

    def test_valid_exact(self):
        afl.cancel(0)

    def test_valid_atleast(self):
        afl.subarray('x')
        afl.subarray('x', 0, 1)


class TestBasicUse(object):

    def test_query(self):
        assert afl.normalize(ARR).query == "normalize(%s)" % ARR.name

    def test_result(self):
        s = afl.normalize(ARR)
        expected = np.ones(4) / 2
        np.testing.assert_array_almost_equal(s.toarray(), expected)

    def test_results_cached(self):
        s = afl.normalize(ARR)
        s.eval() is s.eval()

    def test_eval_nostore(self):
        s = afl.normalize(ARR)
        s.interface = MagicMock()
        s.eval(store=False)
        s.interface._execute_query.assert_called_once_with(s.query)

    def test_eval_output(self):
        s = afl.normalize(ARR)
        out = sdb.new_array()
        result = s.eval(out=out)
        assert result is out
        expected = np.ones(4) / 2
        np.testing.assert_array_almost_equal(s.toarray(), expected)

    def test_access_from_interface(self):
        assert sdb.afl.list("'arrays'").interface is sdb

    def test_interface_acces_cached(self):
        assert sdb.afl is sdb.afl
        assert sdb.afl.normalize is sdb.afl.normalize


class TestFormatOperands(object):

    def test_scalar(self):
        assert _format_operand('x') == 'x'
        assert _format_operand(3) == '3'

    def test_scidbarray_givnen_by_name(self):
        assert _format_operand(ARR) == ARR.name

    def test_expression_given_by_query(self):
        assert _format_operand(afl.normalize(ARR)) == "normalize(%s)" % ARR.name

    def test_expression_given_by_result_name_if_evaluated(self):

        exp = afl.normalize(ARR)
        exp.eval()
        assert _format_operand(exp) == exp.eval().name
