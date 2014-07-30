.. _correlation:

Covariance and correlation matrices
===================================

We can use SciDB's distributed parallel linear algebra operations to
compute covariance and correlation matrices. This demonstrates SciDB-Py's
syntax for performing linear algebra and array arithmetic (with broadcasting).
You can download this script :download:`here<correlation.py>`.

The following example computes the covariance and correlation between
the columns of the matrices X and Y. We break the example up into
a few parts for clarity.

Part 1, set up some example matrices

.. literalinclude:: correlation.py
   :lines: 4-14

Note that, given how y is defined, we expect the correlation matrix to be
approximately equal to::

      array([[1, 1/2, 0],
             [0, 1,   0],
             [0, 0,   0],
             [0, 0,   0],
             [0, 0,   1]])


Part 2, center the example matrices:

.. literalinclude:: correlation.py
   :lines: 16-19


Note that ``X.mean(0)`` computes the sum for each of the five columns of X.
These 5 numbers are subtracted from each row of ``X`` to compute ``XC``.

Part 3, compute the covariance matrix:

.. literalinclude:: correlation.py
   :lines: 21

Part 4, compute the correlation matrix:

.. literalinclude:: correlation.py
   :lines: 23-30

Which prints::

    [[ 1.          0.42008436 -0.01186421]
     [-0.02583465  0.89632943  0.00232589]
     [-0.02296872 -0.00927478  0.00570587]
     [-0.02791049 -0.00367841  0.06882341]
     [-0.01186421 -0.0031508   1.        ]]

The overhead of working interactively with SciDB can make these examples run
somewhat slowly for small problems. But the same code shown here can be
applied to arbitrarily large matrices, and those computations can run in
parallel across a cluster.
