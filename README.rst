========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions|
        | |codecov|
    * - package
      - | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/history_matching/badge/?style=flat
    :target: https://history_matching.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/clorton/history_matching/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/clorton/history_matching/actions

.. |codecov| image:: https://codecov.io/gh/clorton/history_matching/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/clorton/history_matching

.. |commits-since| image:: https://img.shields.io/github/commits-since/clorton/history_matching/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/clorton/history_matching/compare/v0.0.0...main



.. end-badges

History Matching package generated with cookiecutter-pylibrary.

* Free software: MIT license

Installation
============

::

    pip install hm-cc-pylibrary

You can also install the in-development version with::

    pip install https://github.com/clorton/history_matching/archive/main.zip


Documentation
=============


https://history_matching.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
