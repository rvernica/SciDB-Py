from scidbpy.interface import SciDBShimInterface


def test1():
    scidb = SciDBShimInterface('http://localhost:8080')
    session_id = scidb._new_session()
    print scidb._execute_query(session_id,
                               ("CREATE ARRAY A <x: double, err: double> "
                                "[i=0:99,10,0, j=0:99,10,0]"))
    print scidb._execute_query(session_id, "list('arrays')", save='csv')
    print scidb._read_lines(session_id, 5)
    scidb._release_session(session_id)


test1()
