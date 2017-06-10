import scidbpy
import sys
import timeit


def setup(mb):
    cnt = mb * 1024 * 1024 / 8

    db = scidbpy.connect()

    db.iquery(
        'store(build(<x:int64 not null>[i=0:{}], random()), bm)'.format(
            cnt - 1))
    ar = db.iquery('scan(bm)', fetch=True, atts_only=True)
    print("""\
Data size:      {:6.2f} MB
In-memory size: {:6.2f} MB
Number of runs: {:3d}""".format(
      cnt * 8 / 1024. / 1024,
      ar.nbytes / 1024. / 1024,
      runs))

    return db


def cleanup(db):
    db.iquery('remove(bm)')


def download(mb, runs):
    setup = """
import numpy
import scidbpy

db = scidbpy.connect()
id = db._shim(scidbpy.db.Shim.new_session).text
query = 'scan(bm)'
schema = scidbpy.schema.Schema.fromstring('<x:int64 not null>[i]')"""
    stmt = """
db._shim(scidbpy.db.Shim.execute_query,
         id=id,
         query=query,
         save=schema.atts_fmt_scidb)"""
    rt = timeit.Timer(stmt=stmt, setup=setup).timeit(number=runs) / runs
    print("""
Download
--------

SciDB query:      {:6.2f} seconds {:6.2f} MB/second""".format(
      rt, mb / rt))

    setup += stmt
    stmt = """
buf = db._shim(scidbpy.db.Shim.read_bytes, id=id, n=0).content"""
    rt = timeit.Timer(stmt=stmt, setup=setup).timeit(number=runs) / runs
    print("""\
Shim download:    {:6.2f} seconds {:6.2f} MB/second""".format(
      rt, mb / rt))

    setup += stmt

    rt = timeit.Timer(
        stmt="""
data = schema.frombytes(buf)""",
        setup=setup).timeit(number=runs) / runs
    print("""\
NumPy manual:     {:6.2f} seconds {:6.2f} MB/second""".format(
      rt, mb / rt))

    rt = timeit.Timer(
        stmt="""
data = numpy.frombuffer(buf)""",
        setup=setup).timeit(number=runs) / runs
    print("""\
NumPy frombuffer: {:6.2f} seconds {:>6s} MB/second""".format(
      rt, 'NaN' if rt < 0.01 else '{:6.2f}'.format(mb / rt)))

    rt = timeit.Timer(
        stmt="""
db.iquery("scan(bm)", fetch=True, atts_only=True)""",
        setup="""
import scidbpy

db = scidbpy.connect()""").timeit(number=runs) / runs
    print("""\
Total:            {:6.2f} seconds {:6.2f} MB/second""".format(
      rt, mb / rt))


def upload(mb, runs, struct_array, not_null):
    setup = """
import numpy
import scidbpy

db = scidbpy.connect()
ar = db.iquery('{}', fetch=True, atts_only=True){}
schema = scidbpy.schema.Schema.fromstring('<x:int64 {}>[i]')
query = "load(bm, '{fn}', 0, '{fmt}')"
fmt = schema.atts_fmt_scidb
id = db._shim(scidbpy.db.Shim.new_session).text
""".format(
        'scan(bm)' if not_null or not struct_array
        else 'redimension(scan(bm), <x:int64>[i])',
        '' if struct_array else '["x"]',
        'not null' if not_null else '',
        fn='{fn}',
        fmt='{fmt}')
    rt = timeit.Timer(
        stmt="""
upload_data = ar.tobytes()""",
        setup=setup).timeit(number=runs) / runs
    print("""
Upload (NumPy struct_array={}, Schema not_null={})
------

NumPy tobytes:    {:6.2f} seconds {:>6s} MB/second""".format(
      struct_array,
      not_null,
      rt,
      'NaN' if rt < 0.01 else '{:6.2f}'.format(mb / rt)))

    stmt = """
upload_data = schema.tobytes(ar)"""
    rt = timeit.Timer(stmt=stmt, setup=setup).timeit(number=runs) / runs
    print("""\
NumPy manual:     {:6.2f} seconds {:6.2f} MB/second""".format(
      rt, mb / rt))

    setup += stmt
    stmt = """
fn = db._shim(scidbpy.db.Shim.upload, id=id, data=upload_data).text"""
    rt = timeit.Timer(stmt=stmt, setup=setup).timeit(number=runs) / runs
    print("""\
Shim upload:      {:6.2f} seconds {:6.2f} MB/second""".format(
      rt, mb / rt))

    setup += stmt
    stmt = """
db._shim(scidbpy.db.Shim.execute_query,
         id=id,
         query=query.format(fn=fn, fmt=fmt))"""
    rt = timeit.Timer(stmt=stmt, setup=setup).timeit(number=runs) / runs
    print("""\
SciDB query:      {:6.2f} seconds {:6.2f} MB/second""".format(
      rt, mb / rt))

    rt = timeit.Timer(
        stmt="""
db.iquery("load(bm, '{fn}', 0, '{fmt}')",
          upload_data=ar,
          upload_schema=scidbpy.schema.Schema.fromstring(
            "<x:int64 {}>[i]"))
""".format('not null' if not_null else '',
           fn='{fn}',
           fmt='{fmt}'),
        setup="""
import scidbpy

db = scidbpy.connect()
ar = db.iquery('{}', fetch=True, atts_only=True){}
""".format('scan(bm)' if not_null or not struct_array
           else 'redimension(scan(bm), <x:int64>[i])',
           '' if struct_array else '["x"]')).timeit(number=runs) / runs
    print("""\
Total:            {:6.2f} seconds {:6.2f} MB/second""".format(
      rt, mb / rt))


if __name__ == "__main__":
    try:
        mb = int(sys.argv[1])
    except:
        mb = 5                      # MB
    runs = 3

    db = setup(mb)

    download(mb, runs)
    upload(mb, runs, False, True)
    # upload(mb, runs, False, False)  # Not applicable with tobytes
    upload(mb, runs, True, True)
    upload(mb, runs, True, False)

    cleanup(db)
