.. _pointsets:

PointSet
********

Transferring LIDAR point-cloud data from surveying software into 3D modelling software packages.

.. image:: /images/PointSet.png
    :scale: 80%

Element
-------

.. autoclass:: omf.pointset.PointSetElement

Geometry
--------

.. image:: /images/PointSetGeometry.png
    :width: 80%
    :align: center

.. autoclass:: omf.pointset.PointSetGeometry

Data
----

Data is a list of :ref:`data <data>`. For PointSets, only
:code:`location='vertices'` is valid.

Textures
--------

Textures are :ref:`ImageTexture <textures>` mapped to the PointSets.
