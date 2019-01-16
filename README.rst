omf
***

.. image:: https://img.shields.io/pypi/v/omf.svg
    :target: https://pypi.python.org/pypi/omf
    :alt: Latest PyPI version

.. image:: https://readthedocs.org/projects/omf/badge/?version=latest
    :target: http://omf.readthedocs.io/en/latest/
    :alt: Documentation

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/gmggroup/omf/blob/master/LICENSE
    :alt: MIT license

.. image:: https://travis-ci.org/gmggroup/omf.svg?branch=master
    :target: https://travis-ci.org/gmggroup/omf
    :alt: Travis tests

.. image:: https://codecov.io/gh/gmggroup/omf/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/gmggroup/omf
    :alt: Code coverage


Version: 1.0.0

API library for Open Mining Format, a new standard for mining data backed by
the `Global Mining Guidelines Group <https://gmggroup.org>`_.

.. warning::
    **Pre-Release Notice**

    This is a Beta release of the Open Mining Format (OMF) and the associated
    Python API. The storage format and libraries might be changed in
    backward-incompatible ways and are not subject to any SLA or deprecation
    policy.

Why?
----

An open-source serialization format and API library to support data interchange
across the entire mining community.

Scope
-----

This library provides an abstracted object-based interface to the underlying
OMF serialization format, which enables rapid development of the interface while
allowing for future changes under the hood.

Goals
-----

- The goal of Open Mining Format is to standardize data formats across the
  mining community and promote collaboration
- The goal of the API library is to provide a well-documented, object-based
  interface for serializing OMF files

Alternatives
------------

OMF is intended to supplement the many alternative closed-source file formats
used in the mining community.

Connections
-----------

This library makes use of the `properties <https://github.com/seequent/properties>`_
open-source project, which is designed and publicly supported by
`Seequent <https://www.seequent.com>`_.

Installation
------------

To install the repository, ensure that you have
`pip installed <https://pip.pypa.io/en/stable/installing/>`_ and run:

.. code::

    pip install omf

Or from `github <https://github.com/gmggroup/omf>`_:

.. code::

    git clone https://github.com/gmggroup/omf.git
    cd omf
    pip install -e .
