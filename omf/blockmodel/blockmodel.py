"""blockmodel2.py: New Block Model element definitions"""
import numpy as np
import properties

from ..base import ProjectElement


def _shrink_uint(obj, attr):
    arr = getattr(obj, attr)
    assert arr.min() >= 0
    t = np.min_scalar_type(arr.max())
    setattr(obj, attr, arr.astype(t))


class _BlockCount(properties.Array):
    def __init__(self, doc, **kw):
        super().__init__(doc, **kw, dtype=int, shape=(3,))

    def validate(self, instance, value):
        """Check shape and dtype of the count and that items are >= min."""
        value = super().validate(instance, value)
        for item in value:
            if item < 1:
                if instance is None:
                    msg = f"block counts must be >= 1"
                else:
                    cls = instance.__class__.__name__
                    msg = f"{cls}.{self.name} counts must be >= 1"
                raise properties.ValidationError(msg, prop=self.name, instance=instance)
        return value


class _OctreeSubblockCount(_BlockCount):
    def validate(self, instance, value):
        """Check shape and dtype of the count and that items are >= min."""
        value = super().validate(instance, value)
        for item in value:
            l = np.log2(item)
            if np.trunc(l) != l:
                if instance is None:
                    msg = f"octree block counts must be powers of two"
                else:
                    cls = instance.__class__.__name__
                    msg = f"{cls}.{self.name} octree counts must be powers of two"
                raise properties.ValidationError(msg, prop=self.name, instance=instance)
        return value


class _BlockSize(properties.Array):
    def __init__(self, doc, **kw):
        super().__init__(doc, **kw, dtype=float, shape=(3,))

    def validate(self, instance, value):
        """Check shape and dtype of the count and that items are >= min."""
        value = super().validate(instance, value)
        for item in value:
            if item <= 0.0:
                if instance is None:
                    msg = f"block size elements must be > 0.0"
                else:
                    msg = f"{instance.__class__.__name__}.{self.name} elements must be > 0.0"
                raise properties.ValidationError(msg, prop=self.name, instance=instance)
        return value


class _BaseBlockModel(ProjectElement):
    """Basic orientation properties and indexing for all block models"""

    axis_u = properties.Vector3(
        "Vector orientation of u-direction",
        default="X",
        length=1,
    )
    axis_v = properties.Vector3(
        "Vector orientation of v-direction",
        default="Y",
        length=1,
    )
    axis_w = properties.Vector3(
        "Vector orientation of w-direction",
        default="Z",
        length=1,
    )
    corner = properties.Vector3(
        "Corner of the block model relative to Project coordinate reference system",
        default="zero",
    )
    _valid_locations = ("cells", "parent_blocks")

    @properties.validator
    def _validate_axes(self):
        """Check if mesh content is built correctly"""
        if not (
            np.abs(self.axis_u.dot(self.axis_v) < 1e-6)
            and np.abs(self.axis_v.dot(self.axis_w) < 1e-6)
            and np.abs(self.axis_w.dot(self.axis_u) < 1e-6)
        ):
            raise ValueError("axis_u, axis_v, and axis_w must be orthogonal")
        return True

    @property
    def parent_block_count(self):
        """Computed property for number of parent blocks

        This is required for ijk indexing. For tensor and regular block
        models, all the blocks are considered parent blocks.
        """
        raise NotImplementedError()

    def ijk_to_index(self, ijk):
        """Map IJK triples to flat indices for a singoe triple or an array, preseving shape."""
        arr = np.asarray(ijk)
        if arr.dtype.kind not in "ui":
            raise TypeError(f"'ijk' must be integer typed, found {arr.dtype}")
        match arr.shape:
            case (*output_shape, 3):
                shaped = arr.reshape(-1, 3)
            case _:
                raise ValueError(
                    "'ijk' must have 3 elements or be an array with shape (*_, 3)"
                )
        count = self.parent_block_count
        if (shaped < 0).any() or (shaped >= count).any():
            raise IndexError(f"0 <= ijk < ({count[0]}, {count[1]}, {count[2]}) failed")
        indices = np.ravel_multi_index(multi_index=shaped.T, dims=count, order="F")
        if output_shape == ():
            return indices[0]
        else:
            return indices.reshape(output_shape)

    def index_to_ijk(self, index):
        """Map flat indices to IJK triples for a singoe index or an array, preserving shape."""
        arr = np.asarray(index)
        if arr.dtype.kind not in "ui":
            raise TypeError(f"'index' must be integer typed, found {arr.dtype}")
        output_shape = arr.shape + (3,)
        shaped = arr.reshape(-1)
        count = self.parent_block_count
        if (shaped < 0).any() or (shaped >= np.prod(count)).any():
            raise IndexError(f"0 <= index < {np.prod(count)} failed")
        ijk = np.unravel_index(indices=shaped, shape=count, order="F")
        return np.c_[ijk[0], ijk[1], ijk[2]].reshape(output_shape)


class TensorGridBlockModel(_BaseBlockModel):
    """Block model with variable spacing in each dimension.

    Unlike the rest of the block models attributes here can also be on the block vertices.
    """

    schema = "org.omf.v2.element.blockmodel.tensorgrid"

    _valid_locations = ("vertices",) + _BaseBlockModel._valid_locations

    tensor_u = properties.Array(
        "Tensor cell widths, u-direction",
        shape=("*",),
        dtype=float,
    )
    tensor_v = properties.Array(
        "Tensor cell widths, v-direction",
        shape=("*",),
        dtype=float,
    )
    tensor_w = properties.Array(
        "Tensor cell widths, w-direction",
        shape=("*",),
        dtype=float,
    )

    @properties.validator("tensor_u")
    @properties.validator("tensor_v")
    @properties.validator("tensor_w")
    def _validate_tensor(self, change):
        tensor = change["value"]
        if (tensor <= 0.0).any():
            raise properties.ValidationError(
                "Tensor spacings must all be greater than zero",
                prop=change["name"],
                instance=self,
                reason="invalid",
            )

    def _require_tensors(self):
        if self.tensor_u is None or self.tensor_v is None or self.tensor_w is None:
            raise ValueError("tensors haven't been set yet")

    @property
    def parent_block_count(self):
        self._require_tensors()
        return np.array(
            (len(self.tensor_u), len(self.tensor_v), len(self.tensor_w)), dtype=int
        )

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        return np.prod(self.parent_block_count + 1)

    @property
    def num_cells(self):
        """Number of cells"""
        return np.prod(self.parent_block_count)

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        match location:
            case "vertices":
                return self.num_nodes
            case "cells" | "parent_blocks" | "":
                return self.num_cells
            case _:
                raise ValueError(f"unknown location type: {location!r}")


class RegularBlockModel(_BaseBlockModel):
    """Block model with constant spacing in each dimension."""

    schema = "org.omf.v2.elements.blockmodel.regular"

    block_count = _BlockCount("Number of blocks along u, v, and w axes")
    block_size = _BlockSize("Size of blocks in the u, v, and w directions")

    @property
    def num_cells(self):
        """The number of cells, which in this case are always parent blocks."""
        return np.prod(self.parent_block_count)

    @property
    def parent_block_count(self):
        """Number of parent blocks equals number of blocks"""
        return self.block_count

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        match location:
            case "cells" | "parent_blocks" | "":
                return self.num_cells
            case _:
                raise ValueError(f"unknown location type: {location!r}")


class RegularSubblockDefinition:
    """The simplest gridded sub-block definition."""

    count = _BlockCount(
        "The maximum number of sub-blocks inside a parent in each direction."
    )

    def validate_subblocks(self, _corners):
        """Checks the sub-blocks within one parent block."""
        # XXX can we check for overlaps efficiently?


class OctreeSubblockDefinition(RegularSubblockDefinition):
    """Sub-blocks form an octree inside the parent block.

    Cut the parent block in half in all directions to create eight sub-blocks. Repeat that
    division for some or all of those new sub-blocks. Continue doing that until the limit
    on sub-block count is reached or until the sub-blocks accurately model the inputs.

    This definition also allows the lower level cuts to be omitted in one or two axes,
    giving a maximum sub-block count of (16, 16, 4) for example rather than requiring
    all axes to be equal.
    """

    count = _OctreeSubblockCount(
        "The maximum number of sub-blocks inside a parent in each direction."
    )

    def validate_subblocks(self, corners):
        """Checks the sub-blocks within one parent block."""
        super().validate_subblocks(corners)
        # TODO check that blocks lie on the octree


class SubblockedModel(_BaseBlockModel):
    """A model where regular parent blocks are divided into sub-blocks that align with a grid.

    The sub-blocks here must align with a regular grid within the parent block. What that grid
    is and how blocks are generated within it is defined by the sub-block definition.
    """

    schema = "org.omf.v2.elements.blockmodel.subblocked"

    parent_block_count = _BlockCount("Number of parent blocks along u, v, and w axes")
    parent_block_size = _BlockSize(
        "Size of parent blocks in the u, v, and w directions"
    )
    subblock_parent_indices = properties.Array(
        "The parent block IJK index of each sub-block",
        shape=("*", 3),
        dtype=int,
    )
    subblock_corners = properties.Array(
        """The positions of the sub-block corners on the grid within their parent block.

        The columns are (min_i, min_j, min_k, max_i, max_j, max_k). Values must be
        greater than or equal to zero and less than or equal to the maximum number of
        sub-blocks in that axis.

        Sub-blocks must stay within the parent block and should not overlap. Gaps are
        allowed but it will be impossible for 'cell' attributes to assign values to
        those areas.
        """,
        shape=("*", 6),
        dtype=int,
    )
    subblock_definition = properties.Instance(
        "Defines the structure of sub-blocks within each parent block for this model.",
        RegularSubblockDefinition,
    )

    @property
    def subblock_count(self):
        return self.subblock_definition.count

    # XXX add num_cells property and implement location_length

    def _check_lengths(self):
        if len(self.subblock_parent_indices) != len(self.subblock_corners):
            raise ValueError(
                "'subblock_parent_indices' and 'subblock_corners' arrays must be the same length"
            )

    def _check_parent_indices(self):
        indices = self.subblock_parent_indices
        count = self.parent_block_count
        if (indices < 0).any() or (indices >= count).any():
            raise IndexError(
                f"0 <= subblock_parent_indices < ({count[0]}, {count[1]}, {count[2]}) failed"
            )

    def _check_inside_parent(self):
        min_corner = self.subblock_corners[:, :3]
        max_corner = self.subblock_corners[:, 3:]
        count = self.subblock_count
        if min_corner.dtype.kind != "u" and not (0 <= min_corner).all():
            raise IndexError("0 <= min_corner failed")
        if not (min_corner < max_corner).all():
            raise IndexError("min_corner < max_corner failed")
        if not (max_corner <= count).all():
            raise IndexError(
                f"max_corner <= ({count[0]}, {count[1]}, {count[2]}) failed"
            )

    @properties.validator
    def _validate_subblocks(self):
        self._check_lengths()
        self._check_parent_indices()
        self._check_inside_parent()
        # Check corners against the definition.
        # TODO check that sub-blocks in each parent are adjacent or remove that requirement
        indices = self.ijk_to_index(self.subblock_parent_indices)
        for index in np.unique(indices):
            corners = self.subblock_corners[indices == index, :]  # XXX slow
            self.subblock_definition.validate_subblocks(corners)
        # Cast to the smallest unsigned integer type.
        _shrink_uint(self, "subblock_parent_indices")
        _shrink_uint(self, "subblock_corners")


class FreeformSubblockDefinition:
    """Unconstrained free-form sub-block definition.

    Doesn't provide any limitations on or explanation of sub-block positions.
    """

    def validate_subblocks(self, _corners):
        """Checks the sub-blocks within one parent block."""
        # XXX can we check for overlaps efficiently?


class VariableZSubblockDefinition(FreeformSubblockDefinition):
    def validate_subblocks(self, corners):
        """Checks the sub-blocks within one parent block."""
        super().validate_subblocks(corners)
        # TODO check that blocks lie on the octree


class FreeformSubblockedModel(_BaseBlockModel):
    """A model where regular parent blocks are divided into free-form sub-blocks."""

    schema = "org.omf.v2.elements.blockmodel.freeform_subblocked"

    parent_block_count = _BlockCount("Number of parent blocks along u, v, and w axes")
    parent_block_size = _BlockSize(
        "Size of parent blocks in the u, v, and w directions"
    )
    subblock_parent_indices = properties.Array(
        "The parent block IJK index of each sub-block",
        shape=("*", 3),
        dtype=int,
    )
    subblock_corners = properties.Array(
        """The positions of the sub-block corners within their parent block.

        Positions are relative to the parent block, with 0.0 being the minimum side and 1.0 the
        maximum side.

        Sub-blocks must stay within the parent block and should not overlap. Gaps are allowed
        but it will be impossible for 'cell' attributes to assign values to those areas.
        """,
        shape=("*", 6),
        dtype=float,
    )
    subblock_definition = properties.Instance(
        "Defines the structure of sub-blocks within each parent block for this model.",
        FreeformSubblockDefinition,
    )
