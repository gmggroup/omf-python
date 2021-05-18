.. _linesets:

LineSet
*******

Transfer mapped geological contacts from a GIS software package into a 3D modelling software package to help construct a 3D model.

.. image:: /docs/images/LineSet.png
    :scale: 80%

Element
-------

.. image:: /docs/images/LineSetGeometry.png
    :width: 80%
    :align: center

.. autoclass:: omf.lineset.LineSetElement

Data
----

Data is a list of :ref:`data <data>`. For Lines, :code:`location='vertices'`
and :code:`location='segments'` are valid.

