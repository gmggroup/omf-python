.. _blockmodels:

Block Models
************

.. image:: /images/VolumeGrid.png
    :scale: 80%

Element
-------

The `BlockModel` element is used to store all types of block model. Different types
of block model can be described using the `grid` and `subblocks` properties.

.. autoclass:: omf.blockmodel.BlockModel

Block Model Grid
----------------

The blocks of a block model can lie on either a regular or tensor grid. For sub-blocked
models this only applies to the parent blocks.

.. autoclass:: omf.blockmodel.RegularGrid

.. autoclass:: omf.blockmodel.TensorGrid

.. image:: /images/VolumeGridGeometry.png
    :width: 80%
    :align: center

Sub-blocks
----------

.. autoclass:: omf.blockmodel.RegularSubblocks

.. autoclass:: omf.blockmodel.FreeformSubblocks

Attributes
----------

Attributes is a list of :ref:`attributes <attributes>`.

For block models :code:`location='parent_blocks'`, or the backward compatible
:code:`location='cells'`, places attribute values on the parent blocks. There must be a
value for each parent block and ordering is such that as you move down the attribute
array the U index increases fastest, then V, and finally W.

Using :code:`location='vertices'` instead puts the attribute values on the parent block
corners. The ordering is the same.

Sub-blocked models can still have attributes on their parent blocks using the above modes,
or on the sub-blocks using :code:`location='subblocks'`. For sub-blocks the ordering
matches the `corners` array.
