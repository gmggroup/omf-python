.. _projects:

Project
*******

Projects contain a list of :ref:`pointsets`, :ref:`linesets`, :ref:`surfaces`, and
:ref:`volumes`. Projects can be serialized to file using :code:`OMFWriter`:

.. code:: python

    proj = omf.Project()
    ...
    proj.elements = [...]
    ...
    OMFWriter(proj, 'outfile.omf')

For more details on how to build a project, see the :ref:`examples`.

.. autoclass:: omf.base.Project
