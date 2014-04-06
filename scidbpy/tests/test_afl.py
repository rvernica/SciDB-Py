from nose.tools import assert_raises
from mock import MagicMock
import numpy as np

from scidbpy import connect
from scidbpy.afl import _format_operand

sdb = connect()
afl = sdb.afl
ARR = sdb.ones((2, 2))


def test_docstring():
    """Regression test against unintentional docstring changes"""
    expected = "create_array( arrayName, arraySchema )\n\nCreates an array with a given name and schema.\n\nParameters\n----------\n\n    - arrayName: the array name\n    - arraySchema: the array schema of attrs and dims"
    assert afl.create_array.__doc__ == expected


def test_tostring():
    expression = afl.sum(afl.transpose(3))
    assert str(expression) == "sum(transpose(3))"


def test_repr():
    expression = afl.sum(afl.transpose(3))
    assert repr(expression) == "SciDB Expression: <sum(transpose(3))>"


class TestArgumentChecks(object):

    def test_0_args_given_2_expected(self):
        with assert_raises(TypeError) as cm:
            afl.concat()
        assert cm.exception.args[0] == ("concat() takes exactly 2 "
                                        "arguments (0 given)")

    def test_0_args_given_1_expected(self):
        with assert_raises(TypeError) as cm:
            afl.dimensions()
        assert cm.exception.args[0] == ("dimensions() takes exactly 1 "
                                        "argument (0 given)")

    def test_0_args_given_at_least_1_expected(self):
        with assert_raises(TypeError) as cm:
            afl.analyze()
        assert cm.exception.args[0] == ("analyze() takes at least 1 "
                                        "argument (0 given)")

    def test_valid_exact(self):
        afl.cancel(0)

    def test_valid_atleast(self):
        afl.subarray('x')
        afl.subarray('x', 0, 1)


class TestBasicUse(object):

    def test_query(self):
        assert afl.sum(ARR).query == "sum(%s)" % ARR.name

    def test_result(self):
        s = afl.sum(ARR)
        np.testing.assert_array_equal(s.toarray(), [4])

    def test_results_cached(self):
        s = afl.sum(ARR)
        s.eval() is s.eval()

    def test_eval_nostore(self):
        s = afl.sum(ARR)
        s.interface = MagicMock()
        s.eval(store=False)
        s.interface._execute_query.assert_called_once_with(s.query)

    def test_eval_output(self):
        s = afl.sum(ARR)
        out = sdb.new_array()
        result = s.eval(out=out)
        assert result is out
        np.testing.assert_array_equal(s.toarray(), [4])

    def test_access_from_interface(self):
        assert sdb.afl.list("'arrays'").interface is sdb

        np.testing.assert_array_equal(sdb.afl.sum(ARR).toarray(), [4])

    def test_interface_acces_cached(self):
        assert sdb.afl is sdb.afl
        assert sdb.afl.sum is sdb.afl.sum


class TestFormatOperands(object):

    def test_scalar(self):
        assert _format_operand('x') == 'x'
        assert _format_operand(3) == '3'

    def test_scidbarray_givnen_by_name(self):
        assert _format_operand(ARR) == ARR.name

    def test_expression_given_by_query(self):
        assert _format_operand(afl.sum(ARR)) == "sum(%s)" % ARR.name

    def test_expression_given_by_result_name_if_evaluated(self):

        exp = afl.sum(ARR)
        exp.eval()
        assert _format_operand(exp) == exp.eval().name
