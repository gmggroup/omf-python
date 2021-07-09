.. _composites:

Composite Element
*****************

Composite Elements are used to compose multiple other elements into
a single, more complex, grouped object.


Element
-------

.. autoclass:: omf.composite.Composite

Attributes
----------

Attributes is a list of :ref:`attributes <attributes>`. For Composite Elements,
only :code:`location='elements'` is valid. However, attributes may also be
defined on the child :code:`elements`

