How to generate the documentation
--------------------------------

1. Clone repository::

     > git clone git@github.com:Paradigm4/SciDB-Py.git

1. Create ``html`` directory::

     > cd SciDB-Py
     SciDB-Py> mkdir --parents docs/_build/html

1. Clone repository a second time in the ``html`` directory::

     SciDB-Py> cd docs/_build/html
     SciDB-Py/docs/_build/html> git clone git@github.com:Paradigm4/SciDB-Py.git

1. Checkout ``gh-pages`` branch in the ``html`` clone::

     SciDB-Py/docs/_build/html> git checkout gh-pages
     SciDB-Py/docs/_build/html> ls
     _modules
     _sources
     _static
     index.html
     ...

1. Change to ``docs`` directory and run ``make``::

     SciDB-Py/docs/_build/html> cd ../..
     SciDB-Py/docs> make html


1. Change to ``html`` directory and ``commit`` updated docs::

     SciDB-Py/docs> cd _build/html
     SciDB-Py/docs/_build/html> git commit -a -m "Generate docs"
     SciDB-Py/docs/_build/html> git push

1. GitHub updates the website http://paradigm4.github.io/SciDB-Py/
