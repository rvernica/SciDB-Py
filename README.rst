How to generate the documentation
--------------------------------

1. Clone repository::

     > git clone git@github.com:Paradigm4/SciDB-Py.git

2. Create ``_build/html`` directory::

     > cd SciDB-Py
     SciDB-Py> mkdir --parents docs/_build/html

3. Clone repository a second time in the ``html`` directory::

     SciDB-Py> cd docs/_build/html
     SciDB-Py/docs/_build/html> git clone git@github.com:Paradigm4/SciDB-Py.git

4. Checkout ``gh-pages`` branch in the ``html`` clone::

     SciDB-Py/docs/_build/html> git checkout gh-pages
     SciDB-Py/docs/_build/html> ls
     _modules
     _sources
     _static
     index.html
     ...

5. Change to ``docs`` directory and run ``make html``::

     SciDB-Py/docs/_build/html> cd ../..
     SciDB-Py/docs> make html


6. Change to ``html`` directory and commit the updated docs::

     SciDB-Py/docs> cd _build/html
     SciDB-Py/docs/_build/html> git commit --all --message="Re-generate docs"
     SciDB-Py/docs/_build/html> git push

7. GitHub updates the website http://paradigm4.github.io/SciDB-Py/
