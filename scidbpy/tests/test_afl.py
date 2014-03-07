from nose.tools import assert_raises
import numpy as np

from scidbpy import afl, interface

sdb = interface.SciDBShimInterface('http://192.168.56.101:8080')
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


class TestFormatOperands(object):

    def test_scalar(self):
        assert afl._format_operand('x') == 'x'
        assert afl._format_operand(3) == '3'

    def test_scidbarray_givnen_by_name(self):
        assert afl._format_operand(ARR) == ARR.name

    def test_expression_given_by_query(self):
        assert afl._format_operand(afl.sum(ARR)) == "sum(%s)" % ARR.name

    def test_expression_given_by_result_name_if_evaluated(self):

        exp = afl.sum(ARR)
        exp.eval()
        assert afl._format_operand(exp) == exp.eval().name
