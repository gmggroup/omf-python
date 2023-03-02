import numpy as np
import properties

from ._properties import (
    BlockCount,
    BlockSize,
    OctreeSubblockCount,
    SubBlockCount,
    TensorArray,
)


class _BaseBlockModelDefinition(properties.HasProperties):
    axis_u = properties.Vector3("Vector orientation of u-direction", default="X", length=1)
    axis_v = properties.Vector3("Vector orientation of v-direction", default="Y", length=1)
    axis_w = properties.Vector3("Vector orientation of w-direction", default="Z", length=1)
    origin = properties.Vector3(
        "Minimum corner of the block model relative to Project coordinate reference system",
        default="zero",
    )
    block_count = None

    @properties.validator
    def _validate_axes(self):
        """Check if mesh content is built correctly"""
        if not (
            np.abs(self.axis_u.dot(self.axis_v) < 1e-6)
            and np.abs(self.axis_v.dot(self.axis_w) < 1e-6)
            and np.abs(self.axis_w.dot(self.axis_u) < 1e-6)
        ):
            raise ValueError("axis_u, axis_v, and axis_w must be orthogonal")

    def ijk_to_index(self, ijk):
        """Map IJK triples to flat indices for a singoe triple or an array, preseving shape."""
        if self.block_count is None:
            raise ValueError("block_count is not set")
        arr = np.asarray(ijk)
        if arr.dtype.kind not in "ui":
            raise TypeError(f"'ijk' must be integer typed, found {arr.dtype}")
        match arr.shape:
            case (*output_shape, 3):
                shaped = arr.reshape(-1, 3)
            case _:
                raise ValueError("'ijk' must have 3 elements or be an array with shape (*_, 3)")
        count = self.block_count
        if (shaped < 0).any() or (shaped >= count).any():
            raise IndexError(f"0 <= ijk < ({count[0]}, {count[1]}, {count[2]}) failed")
        indices = np.ravel_multi_index(multi_index=shaped.T, dims=count, order="F")
        if output_shape == ():
            return indices[0]
        else:
            return indices.reshape(output_shape)

    def index_to_ijk(self, index):
        """Map flat indices to IJK triples for a singoe index or an array, preserving shape."""
        if self.block_count is None:
            raise ValueError("block_count is not set")
        arr = np.asarray(index)
        if arr.dtype.kind not in "ui":
            raise TypeError(f"'index' must be integer typed, found {arr.dtype}")
        output_shape = arr.shape + (3,)
        shaped = arr.reshape(-1)
        count = self.block_count
        if (shaped < 0).any() or (shaped >= np.prod(count)).any():
            raise IndexError(f"0 <= index < {np.prod(count)} failed")
        ijk = np.unravel_index(indices=shaped, shape=count, order="F")
        return np.c_[ijk[0], ijk[1], ijk[2]].reshape(output_shape)


class RegularBlockModelDefinition(_BaseBlockModelDefinition):
    """Defines the block structure of a regular block model.

    If used on a sub-blocked model then everything here applies to the parent blocks only.
    """

    schema = "org.omf.v2.blockmodeldefinition.regular"

    block_count = BlockCount("Number of blocks in each of the u, v, and w directions.")
    block_size = BlockSize("Size of blocks in the u, v, and w directions.")


class TensorBlockModelDefinition(_BaseBlockModelDefinition):
    """Defines the block structure of a tensor grid block model."""

    schema = "org.omf.v2.blockmodeldefinition.tensor"

    tensor_u = TensorArray("Tensor cell widths, u-direction")
    tensor_v = TensorArray("Tensor cell widths, v-direction")
    tensor_w = TensorArray("Tensor cell widths, w-direction")

    def _tensors(self):
        yield self.tensor_u
        yield self.tensor_v
        yield self.tensor_w

    @property
    def block_count(self):
        """The block count is derived from the tensors here."""
        c = tuple(None if t is None else len(t) for t in self._tensors())
        if None in c:
            return None
        return np.array(c, dtype=int)


class RegularSubblockDefinition(properties.HasProperties):
    """The simplest gridded sub-block definition."""

    schema = "org.omf.v2.subblockdefinition.regular"

    subblock_count = SubBlockCount("The maximum number of sub-blocks inside a parent in each direction.")


class OctreeSubblockDefinition(RegularSubblockDefinition):
    """Sub-blocks form an octree inside the parent block.

    Cut the parent block in half in all directions to create eight sub-blocks. Repeat that
    division for some or all of those new sub-blocks. Continue doing that until the limit
    on sub-block count is reached or until the sub-blocks accurately model the inputs.

    This definition also allows the lower level cuts to be omitted in one or two axes,
    giving a maximum sub-block count of (16, 16, 4) for example rather than requiring
    all axes to be equal.
    """

    schema = "org.omf.v2.subblockdefinition.octree"

    subblock_count = OctreeSubblockCount("The maximum number of sub-blocks inside a parent in each direction.")


class FreeformSubblockDefinition(properties.HasProperties):
    """Unconstrained free-form sub-block definition.

    Provides no limitations on or explanation of sub-block positions.
    """

    schema = "org.omf.v2.subblockdefinition.freeform"


class VariableHeightSubblockDefinition(FreeformSubblockDefinition):
    """Defines sub-blocks on a grid in the U and V directions but variable in the W direction.

    A single sub-block covering the whole parent block is also valid. Sub-blocks should not
    overlap.

    Note: these constraints on sub-blocks are not checked during validation.
    """

    schema = "org.omf.v2.subblockdefinition.variableheight"

    subblock_count_u = properties.Integer("Number of sub-blocks in the u-direction", min=1, max=65535)
    subblock_count_v = properties.Integer("Number of sub-blocks in the v-direction", min=1, max=65535)
    minimum_size_w = properties.Float("Minimum size of sub-blocks in the z-direction", min=0.0)
