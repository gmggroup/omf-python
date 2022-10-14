.. _linesets:

LineSet
*******

Transfer mapped geological contacts from a GIS software package into a 3D modelling software package to help construct a 3D model.

.. image:: /images/LineSet.png
    :scale: 80%

Element
-------

.. autoclass:: omf.lineset.LineSetElement

Geometry
--------

.. image:: /images/LineSetGeometry.png
    :width: 80%
    :align: center

.. autoclass:: omf.lineset.LineSetGeometry

Data
----

Data is a list of :ref:`data <data>`. For Lines, :code:`location='vertices'`
and :code:`location='segments'` are valid.
