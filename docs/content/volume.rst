.. _volumes:

Volume
******

Transferring a block model from Resource Estimation software into Mine planning software.

.. image:: /images/VolumeGrid.png
    :scale: 80%

Element
-------

.. autoclass:: omf.volume.VolumeElement

Geometry
--------

.. image:: /images/VolumeGridGeometry.png
    :width: 80%
    :align: center

.. autoclass:: omf.volume.VolumeGridGeometry

Data
----

Data is a list of :ref:`data <data>`. For Volumes, :code:`location='vertices'`
and :code:`location='cells'` are valid.
