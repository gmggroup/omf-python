"""blockmodel/models.py: block model element and grids."""
import numpy as np
import properties

from ..base import BaseModel, ProjectElement
from .index import ijk_to_index, index_to_ijk
from .subblock_check import subblock_check
from .subblocks import FreeformSubblocks, RegularSubblocks

__all__ = ["BlockModel", "RegularGrid", "TensorGrid"]


class RegularGrid(BaseModel):
    """Describes a regular grid of blocks with equal sizes."""

    schema = "org.omf.v2.elements.blockmodel.grid.regular"

    block_count = properties.Array("Number of blocks in each of the u, v, and w directions.", dtype=int, shape=(3,))
    block_size = properties.Vector3("Size of blocks in the u, v, and w directions.", default=lambda: (1.0, 1.0, 1.0))

    @properties.validator("block_count")
    def _validate_block_count(self, change):
        for item in change["value"]:
            if item < 1:
                raise properties.ValidationError("block counts must be >= 1", prop=change["name"], instance=self)

    @properties.validator("block_size")
    def _validate_block_size(self, change):
        for item in change["value"]:
            if item <= 0.0:
                raise properties.ValidationError("block sizes must be > 0.0", prop=change["name"], instance=self)


class TensorGrid(BaseModel):
    """Describes a grid with varied spacing in each direction."""

    schema = "org.omf.v2.elements.blockmodel.grid.tensor"

    tensor_u = properties.Array("Tensor cell widths, u-direction", dtype=float, shape=("*",))
    tensor_v = properties.Array("Tensor cell widths, v-direction", dtype=float, shape=("*",))
    tensor_w = properties.Array("Tensor cell widths, w-direction", dtype=float, shape=("*",))

    @properties.validator("tensor_u")
    def _validate_tensor_u(self, change):
        self._validate_tensor(change)

    @properties.validator("tensor_v")
    def _validate_tensor_v(self, change):
        self._validate_tensor(change)

    @properties.validator("tensor_w")
    def _validate_tensor_w(self, change):
        self._validate_tensor(change)

    def _validate_tensor(self, change):
        if len(change["value"]) == 0:
            raise properties.ValidationError("tensor array may not be empty", prop=change["name"], instance=self)
        for item in change["value"]:
            if item <= 0.0:
                raise properties.ValidationError("tensor sizes must be > 0.0", prop=change["name"], instance=self)

    @property
    def block_count(self):
        """The block count is derived from the tensors here."""
        counts = []
        for tensor in self.tensor_u, self.tensor_v, self.tensor_w:
            if tensor is None:
                return None
            counts.append(len(tensor))
        return np.array(counts, dtype=int)


class BlockModel(ProjectElement):
    """A block model with optional sub-blocks.

    The position and orientation are defined by the `origin`, `axis_u`, `axis_v`, `axis_w`
    attributes, while the block layout and size are defined by the `grid` attribute.

    Sub-blocks are stored in the `subblocks` attribute. Use :code:`None` if there are no
    sub-blocks, :class:`omf.blockmodel.RegularSubblocks` if the sub-blocks lie on a regular
    grid within each parent block, or :class:`omf.blockmodel.FreeformSubblocks` if the
    sub-blocks are not constrained.
    """

    schema = "org.omf.v2.elements.blockmodel"
    _valid_locations = ("parent_blocks", "subblocks", "vertices", "cells")

    origin = properties.Vector3(
        "Minimum corner of the block model relative to Project coordinate reference system",
        default="zero",
    )
    axis_u = properties.Vector3("Vector orientation of u-direction", default="X", length=1)
    axis_v = properties.Vector3("Vector orientation of v-direction", default="Y", length=1)
    axis_w = properties.Vector3("Vector orientation of w-direction", default="Z", length=1)
    grid = properties.Union(
        """Describes the grid that the blocks occupy, either regular or tensor""",
        props=[RegularGrid, TensorGrid],
        default=RegularGrid,
    )
    subblocks = properties.Union(
        """Optional sub-blocks""",
        props=[FreeformSubblocks, RegularSubblocks],
        required=False,
    )

    @properties.validator
    def _validate(self):
        if not (
            np.abs(self.axis_u.dot(self.axis_v) < 1e-6)
            and np.abs(self.axis_v.dot(self.axis_w) < 1e-6)
            and np.abs(self.axis_w.dot(self.axis_u) < 1e-6)
        ):
            raise properties.ValidationError("axis_u, axis_v, and axis_w must be orthogonal", instance=self)
        subblock_check(self)

    @property
    def block_count(self):
        """Number of blocks in each of the u, v, and w directions.

        Equivalent to `block_model.definition.block_count`.
        """
        return self.grid.block_count

    @property
    def num_parent_blocks(self):
        """The number of cells."""
        return np.prod(self.grid.block_count)

    @property
    def num_parent_vertices(self):
        """Number of nodes or vertices."""
        count = self.grid.block_count
        return None if count is None else np.prod(count + 1)

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        if location == "vertices":
            return self.num_parent_vertices
        if location == "subblocks":
            return None if self.subblocks is None else self.subblocks.num_subblocks
        return self.num_parent_blocks

    def ijk_to_index(self, ijk):
        """Map IJK triples to flat indices for a single triple or an array, preserving shape."""
        if self.grid.block_count is None:
            raise ValueError("block count is not yet known")
        return ijk_to_index(self.block_count, ijk)

    def index_to_ijk(self, index):
        """Map flat indices to IJK triples for a single index or an array, preserving shape."""
        if self.block_count is None:
            raise ValueError("block count is not yet known")
        return index_to_ijk(self.block_count, index)
