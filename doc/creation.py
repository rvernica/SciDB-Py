from __future__ import print_function

import numpy as np
np.random.seed(42)

from scidbpy import connect
sdb = connect()

X = np.arndom.random((5, 4))
Xsdb = sdb.from_array(X)


from scipy.sparse import coo_matrix
X = np.random.random((10, 10))
X[X < 0.9] = 0  # make array sparse
Xcoo = coo_matrix(X)
Xsdb = sdb.from_sparse(Xcoo)

# Create a 10x10 array of double-precision zeros:
A = sdb.zeros((10, 10))
# Create a 10x10 array of 64-bit signed integer ones:
A = sdb.ones((10, 10), dtype='int64')

# Create a 10x10 array of numbers between -1 and 2 (inclusive)
#    sampled from a uniform random distribution.
A = sdb.random((10, 10), lower=-1, upper=2)

# Create a 10x10 array of uniform random integers between 0 and 10
#  (inclusive of 0, non-inclusive of 10)
A = sdb.randint((10, 10), lower=0, upper=10)

# Create a vector of ten integers, counting up from zero
A = sdb.arange(10)

# Create a vector of 5 equally spaced numbers between 1 and 10,
# including the endpoints:
A = sdb.linspace(1, 10, 5)

# Create a 10x10 sparse, double-precision-valued identity matrix:
A = sdb.identity(10, dtype='double', sparse=True)

# remove A if it already exists
if "A" in sdb.list_arrays():
    sdb.query("remove(A)")

# create an array named 'A' on the server
sdb.query("store(build(<v:double>[i=1:10,10,0,j=1:5,5,0],i+j),A)")

# create a Python object pointing to this array
A = sdb.wrap_array("A")

X = sdb.random(10, persistent=False)  # default
print(X.name in sdb.list_arrays())  # True
X.reap()
print(X.name in sdb.list_arrays())  # False

with connect(url) as sdb:
    X = sdb.random(10)
# deleted here
