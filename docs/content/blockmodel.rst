.. _blockmodels:

Block Models
************

.. image:: /images/VolumeGrid.png
    :scale: 80%

Element
-------

.. image:: /images/VolumeGridGeometry.png
    :width: 80%
    :align: center

.. autoclass:: omf.blockmodel.TensorGridBlockModel

.. autoclass:: omf.blockmodel.RegularBlockModel

.. autoclass:: omf.blockmodel.RegularSubBlockModel

.. autoclass:: omf.blockmodel.OctreeSubBlockModel

.. autoclass:: omf.blockmodel.ArbitrarySubBlockModel

Attributes
----------

Attributes is a list of :ref:`attributes <attributes>`. For block models,
:code:`location='parent_blocks'` and :code:`location='sub_blocks'` are valid.
