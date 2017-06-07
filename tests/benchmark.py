import scidbpy
import sys
import timeit

if __name__ == "__main__":
    try:
        mb = int(sys.argv[1])
    except:
        mb = 5                      # MB

    cnt = mb * 1024 * 1024 / 8
    runs = 3

    db = scidbpy.connect()

    db.iquery(
        'store(build(<x:int64 not null>[i=0:{}], random()), bm)'.format(
            cnt - 1))
    ar = db.iquery('scan(bm)', fetch=True, atts_only=True)

    print("""\
Data size:      {:6.2f} MB
In-memory size: {:6.2f} MB
Object size:    {:6.2f} MB
Number of runs: {:3d}""".format(
        cnt * 8 / 1024. / 1024,
        ar.nbytes / 1024. / 1024,
        sys.getsizeof(ar) / 1024. / 1024,
        runs))

    rt = timeit.Timer(
        stmt="db.iquery('scan(bm)', fetch=True, atts_only=True)",
        setup="import scidbpy; db = scidbpy.connect()").timeit(
            number=runs) / runs

    print("""
Download time:              {:6.2f} seconds {:6.2f} MB/second""".format(
        rt, mb / rt))

    # NumPy single array, not null schema
    rt = timeit.Timer(
        stmt=("db.iquery(\"" +
              "load(bm, '{fn}', 0, '{fmt}')" +
              "\", upload_data=ar['x'])"),
        setup=("import scidbpy; " +
               "db = scidbpy.connect(); " +
               "ar = db.iquery('scan(bm)', fetch=True, atts_only=True)")
    ).timeit(number=runs) / runs

    print("""\
Upload (single) time:       {:6.2f} seconds {:6.2f} MB/second""".format(
        rt, mb / rt))

#     # NumPy single array, null schema
#     rt = timeit.Timer(
#         stmt=("db.iquery(\"" +
#               "load(bm, '{fn}', 0, '{fmt}')" +
#               "\", upload_data=ar['x'], " +
#               "upload_schema=scidbpy.schema.Schema.fromstring('" +
#               "<x:int64>[i]" +
#               "'))"),
#         setup=("import scidbpy; " +
#                "db = scidbpy.connect(); " +
#                "ar = db.iquery('scan(bm)', fetch=True, atts_only=True)")
#     ).timeit(number=runs) / runs

#     print("""\
# Upload (single/null) time:  {:6.2f} seconds {:6.2f} MB/second""".format(
#         rt, mb / rt))

    # NumPy structured array, not null schema
    rt = timeit.Timer(
        stmt=("db.iquery(\"" +
              "load(bm, '{fn}', 0, '{fmt}')" +
              "\", upload_data=ar)"),
        setup=("import scidbpy; " +
               "db = scidbpy.connect(); " +
               "ar = db.iquery('scan(bm)', fetch=True, atts_only=True)")
    ).timeit(number=runs) / runs

    print("""\
Upload (struct.) time:      {:6.2f} seconds {:6.2f} MB/second""".format(
        rt, mb / rt))

    # NumPy structured array, null schema
    rt = timeit.Timer(
        stmt=("db.iquery(\"" +
              "load(bm, '{fn}', 0, '{fmt}')" +
              "\", upload_data=ar, " +
              "upload_schema=scidbpy.schema.Schema.fromstring('" +
              "<x:int64>[i]" +
              "'))"),
        setup=("import scidbpy; " +
               "db = scidbpy.connect(); " +
               "ar = db.iquery('redimension(scan(bm), <x:int64>[i])', " +
               "fetch=True, atts_only=True)")
    ).timeit(number=runs) / runs

    print("""\
Upload (struct./null) time: {:6.2f} seconds {:6.2f} MB/second""".format(
        rt, mb / rt))

    db.iquery('remove(bm)')
