import numpy as np
np.random.seed(42)

#  connect to the database
from scidbpy import connect
sdb = connect()

# Two small arrays, each with 1000 rows, and with 5 and 3 columns
x = np.random.random((1000, 5))
y = np.column_stack((x[:, 0] * 2, x[:, 1] + x[:, 0] / 2., x[:, 4]))

# Transfer to the database. All future computation happens there.
X = sdb.from_array(x)
Y = sdb.from_array(y)

# Subtract the column means from X using broadcasting:
XC = X - X.mean(0)
# Similarly subtract the column means from Y:
YC = Y - Y.mean(0)

COV = sdb.dot(XC.T, YC) / (X.shape[0] - 1)

# Column vector with column standard deviations of X matrix:
xsd = X.std(0).reshape((5, 1))
# Row vector with column standard deviations of Y matrix:
ysd = Y.std(0).reshape((1, 3))
# Their outer product:
outersd = sdb.dot(xsd, ysd)
COR = COV / outersd
print(COR.toarray())
