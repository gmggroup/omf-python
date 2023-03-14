.. _blockmodels:

Block Models
************

.. image:: /images/VolumeGrid.png
    :scale: 80%

Element
-------

The `BlockModel` element is used to store all types of block model. Sub-types are
described by the block model definition, the presence or absense of sub-blocks, the
type of those sub-blocks, and finally by the sub-block mode.

.. autoclass:: omf.blockmodel.BlockModel

Block Model Definitions
-----------------------

These are two choices for the block model definition: regular and tensor. In a regular
model parent blocks are all the same size and shape.

.. autoclass:: omf.blockmodel.RegularBlockModelDefinition

A tensor model has varying spacings along each axis, allowing for more detail in some
areas without the added complexity of sub-blocking. This is more common in some types
of geo-physical grids. These models don't normally have sub-blocks.

.. image:: /images/VolumeGridGeometry.png
    :width: 80%
    :align: center

.. autoclass:: omf.blockmodel.TensorBlockModelDefinition

Regular Sub-blocks
------------------

Regular sub-blocks must be aligned to some lower-level regular grid within the parent
block. Sub-blocks must stay within the parent, must not overlap, and must have size
greater than zero in all directions. Gaps are allowed but it will be impossible to
place any attribute values in those gaps. If a parent is not sub-blocked that should
be represented as a sub-block thats cover the entire parent block.

Sub-blocks can be further restrictied using the `OctreeSubblockDefinition`. This
requires the numbers of sub-blocks grid size in each direction to be a power of two
and for the sub-blocks to conform to an octree structure within each parent.

.. autoclass:: omf.blockmodel.RegularSubblocks

.. autoclass:: omf.blockmodel.SubblockModeOctree

.. autoclass:: omf.blockmodel.SubblockModeFull

Free-form Sub-blocks
--------------------

Free-form sub-blocks are similar to regular but don't follow any structure or grid
within their parent blocks. Sub-blocks must stay within the parent and must have size
greater than zero in all directions. They probably shouldn't overlap but that isn't
checked.

.. autoclass:: omf.blockmodel.FreeformSubblocks

Attributes
----------

Attributes is a list of :ref:`attributes <attributes>`. For block models,
:code:`location='parent_blocks'`, :code:`location='vertices'`, and :code:`location='cells'`
are valid.
