"""blockmodel/models.py: block model elements."""
import numpy as np
import properties

from ..base import BaseModel, ProjectElement
from ._utils import ijk_to_index, index_to_ijk
from .freeform_subblocks import FreeformSubblocks
from .regular_subblocks import RegularSubblocks

__all__ = ["BlockModel", "RegularBlockModelDefinition", "TensorBlockModelDefinition"]


class RegularBlockModelDefinition(BaseModel):
    """Defines the block structure of a regular block model.

    If used on a sub-blocked model then everything here applies to the parent blocks only.
    """

    schema = "org.omf.v2.elements.blockmodel.regular"

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


class TensorBlockModelDefinition(BaseModel):
    """Defines the block structure of a tensor grid block model."""

    schema = "org.omf.v2.elements.blockmodel.tensor"

    tensor_u = properties.Array("Tensor cell widths, u-direction", dtype=float, shape=("*",))
    tensor_v = properties.Array("Tensor cell widths, v-direction", dtype=float, shape=("*",))
    tensor_w = properties.Array("Tensor cell widths, w-direction", dtype=float, shape=("*",))

    @properties.validator("tensor_u")
    @properties.validator("tensor_v")
    @properties.validator("tensor_w")
    def _validate_tensor(self, change):
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
    """A block model, details are in the definition and sub-blocks attributes."""

    schema = "org.omf.v2.elements.blockmodel"
    _valid_locations = ("parent_blocks", "vertices", "cells")

    origin = properties.Vector3(
        "Minimum corner of the block model relative to Project coordinate reference system",
        default="zero",
    )
    axis_u = properties.Vector3("Vector orientation of u-direction", default="X", length=1)
    axis_v = properties.Vector3("Vector orientation of v-direction", default="Y", length=1)
    axis_w = properties.Vector3("Vector orientation of w-direction", default="Z", length=1)
    definition = properties.Union(
        """Block model definition, describing either a regular or tensor-based block layout.""",
        props=[RegularBlockModelDefinition, TensorBlockModelDefinition],
        default=RegularBlockModelDefinition,
    )
    subblocks = properties.Union(
        """Optional sub-block details.

        If this is `None` then there are no sub-blocks. Otherwise it can be a `FreeformSubblocks`
        or `RegularSubblocks` object to define different types of sub-blocks.
        """,
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
        if self.subblocks is not None:
            self.subblocks.validate_subblocks(self)

    @property
    def block_count(self):
        """Number of blocks in each of the u, v, and w directions.

        Equivalent to `block_model.definition.block_count`.
        """
        return self.definition.block_count

    @property
    def num_parent_blocks(self):
        """The number of cells."""
        return np.prod(self.definition.block_count)

    @property
    def num_parent_vertices(self):
        """Number of nodes or vertices."""
        count = self.definition.block_count
        return None if count is None else np.prod(count + 1)

    @property
    def num_cells(self):
        """The number of cells."""
        return self.num_parent_blocks if self.subblocks is None else self.subblocks.num_subblocks

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        if location == "vertices":
            return self.num_parent_vertices
        if location == "cells" and self.subblocks is not None:
            return self.subblocks.num_subblocks
        return self.num_parent_blocks

    def ijk_to_index(self, ijk):
        """Map IJK triples to flat indices for a single triple or an array, preserving shape."""
        if self.definition.block_count is None:
            raise ValueError("block count is not yet known")
        return ijk_to_index(self.definition.block_count, ijk)

    def index_to_ijk(self, index):
        """Map flat indices to IJK triples for a single index or an array, preserving shape."""
        if self.definition.block_count is None:
            raise ValueError("block count is not yet known")
        return index_to_ijk(self.definition.block_count, index)
