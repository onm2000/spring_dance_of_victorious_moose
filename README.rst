========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-binding_prediction/badge/?style=flat
    :target: https://readthedocs.org/projects/python-binding_prediction
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/ehthiede/python-binding_prediction.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/ehthiede/python-binding_prediction

.. |codecov| image:: https://codecov.io/gh/ehthiede/python-binding_prediction/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/ehthiede/python-binding_prediction

.. |commits-since| image:: https://img.shields.io/github/commits-since/ehthiede/python-binding_prediction/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ehthiede/python-binding_prediction/compare/v0.0.0...master



.. end-badges

An example package. Generated with cookiecutter-pylibrary.

* Free software: MIT license

Installation
============

::

    pip install binding-prediction

You can also install the in-development version with::

    pip install https://github.com/ehthiede/python-binding_prediction/archive/master.zip


Documentation
=============


https://python-binding_prediction.readthedocs.io/


Development
===========

To run the all tests run::

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
