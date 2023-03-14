.. _blockmodels:

Block Models
************

.. image:: /images/VolumeGrid.png
    :scale: 80%

Elements
--------

.. image:: /images/VolumeGridGeometry.png
    :width: 80%
    :align: center

.. autoclass:: omf.blockmodel.RegularBlockModel

.. autoclass:: omf.blockmodel.TensorGridBlockModel

.. autoclass:: omf.blockmodel.SubblockedModel

.. autoclass:: omf.blockmodel.FreeformSubblockedModel


Block Model Definitions
-----------------------

These classes are used as part of the block model elements to define the position
and size of the model.

.. autoclass:: omf.blockmodel.RegularBlockModelDefinition

.. autoclass:: omf.blockmodel.TensorBlockModelDefinition

Sub-block Definitions
---------------------

These classes are used to define the structure of sub-blocks within a parent block.

.. autoclass:: omf.blockmodel.RegularSubblockDefinition

.. autoclass:: omf.blockmodel.OctreeSubblockDefinition

.. autoclass:: omf.blockmodel.FreeformSubblockDefinition

.. autoclass:: omf.blockmodel.VariableHeightSubblockDefinition

Attributes
----------

Attributes is a list of :ref:`attributes <attributes>`. For block models,
:code:`location='parent_blocks'` and :code:`location='sub_blocks'` are valid.
