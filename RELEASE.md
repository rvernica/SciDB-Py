Releasing a new version
=======================

1. Make sure the test suite passes
2. Update ``CHANGES.txt`` and ``doc/whats_new.rst`` to document
   changes. Update version references in ``README.md``
3. Update ``__version__`` in ``__init__.py``
4. Update the version number in ``doc/conf.py``
5. Run ``git clean -fxd`` to remove any non-committed files
6. Run

    python setup.py sdist --format=gztar

    and make sure that the generated file is good to go
    by going inside ``dist``, expanding the tar file, and
    running the tests with

    python setup.py test

7. Re-run ``git clean -fxd``
8. Commit the changes to CHANGES.md and setup.py:
   git commit -m "Preparing release <version>"

9. Tag commit with ``v<version>``
10. Change version in ``__init__.py`` back to a ``.dev`` entry.
    Add a new entry to CHANGES.md
11. Commit with message:
    git commit -m "Back to development: <next_version>"
12. Release the commit:
    git co -v<version>
    git clean -fxd
    python setup.py sdist --format=gztar upload
13. Fast-forward the ``stable`` branch, and push to GitHub.
14. Add the new version to the build-list on readthedocs.
