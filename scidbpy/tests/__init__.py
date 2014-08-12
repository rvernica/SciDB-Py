# License: Simplified BSD, 2014
# See LICENSE.txt for more information

from .. import connect
sdb = connect()

# NOTE: this needs to be explicitly imported in each test module,
#      or pytest won't run it


def teardown_function(function):
    sdb.reap()


class TestBase(object):

    def teardown_method(self, method):
        sdb.reap()
