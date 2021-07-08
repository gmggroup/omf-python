.. _pointsets:

PointSet
********

Transfering LIDAR point-cloud data from surveying software into 3D modelling software packages.

.. image:: /images/PointSet.png
    :scale: 80%

Element
-------

.. image:: /images/PointSetGeometry.png
    :width: 80%
    :align: center

.. autoclass:: omf.pointset.PointSetElement

Attributes
----------

Attributes is a list of :ref:`attributes <attributes>`. For PointSets, only
:code:`location='vertices'` is valid.

Textures
--------

Textures are :ref:`ImageTexture <textures>` mapped to the PointSets.
