.. _composites:

Composite Element
*****************

Composite Elements are used to compose multiple other elements into
a single, more complex, grouped object.


Element
-------

.. autoclass:: omf.composite.CompositeElement

Data
----

Data is a list of :ref:`data <data>`. For Composite Elements,
only :code:`location='elements'` is valid. However, Data may also be
defined on the child :code:`elements`

