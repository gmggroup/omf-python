"""blockmodel/models.py: block model elements."""
import numpy as np
import properties

from ..attribute import Array, ArrayInstanceProperty
from ..base import ProjectElement
from .definition import (
    FreeformSubblockDefinition,
    OctreeSubblockDefinition,
    RegularBlockModelDefinition,
    RegularSubblockDefinition,
    TensorBlockModelDefinition,
    VariableHeightSubblockDefinition,
)
from ._subblock_check import check_subblocks


def _shrink_uint(arr):
    assert arr.array.dtype.kind in "ui"
    if arr.array.min() >= 0:
        arr.array = arr.array.astype(np.min_scalar_type(arr.array.max()))


class RegularBlockModel(ProjectElement):
    """A block model with fixed size blocks on a regular grid and no sub-blocks."""

    schema = "org.omf.v2.elements.blockmodel.regular"
    _valid_locations = ("cells", "parent_blocks")

    definition = properties.Instance(
        "Block model definition",
        RegularBlockModelDefinition,
        default=RegularBlockModelDefinition,
    )

    @property
    def num_cells(self):
        """The number of cells, which in this case are always parent blocks."""
        return np.prod(self.definition.block_count)

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        return self.num_cells


class TensorGridBlockModel(ProjectElement):
    """A block model with variable spacing in all directions and no sub-blocks."""

    schema = "org.omf.v2.element.blockmodel.tensorgrid"
    _valid_locations = ("vertices", "cells", "parent_blocks")

    definition = properties.Instance(
        "Block model definition, including the tensor arrays",
        TensorBlockModelDefinition,
        default=TensorBlockModelDefinition,
    )

    @property
    def num_cells(self):
        """The number of cells."""
        return np.prod(self.definition.block_count)

    @property
    def num_nodes(self):
        """Number of nodes or vertices."""
        count = self.definition.block_count
        return None if count is None else np.prod(count + 1)

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        return self.num_nodes if location == "vertices" else self.num_cells


class SubblockedModel(ProjectElement):
    """A regular block model with sub-blocks that align with a lower-level grid."""

    schema = "org.omf.v2.elements.blockmodel.subblocked"
    _valid_locations = ("cells", "parent_blocks")

    definition = properties.Instance(
        "Block model definition, for the parent blocks",
        RegularBlockModelDefinition,
        default=RegularBlockModelDefinition,
    )
    subblock_definition = properties.Union(
        "Defines the structure of sub-blocks within each parent block.",
        props=[
            properties.Instance("", RegularSubblockDefinition),
            properties.Instance("", OctreeSubblockDefinition),
        ],
        default=RegularSubblockDefinition,
    )
    subblock_parent_indices = ArrayInstanceProperty(
        "The parent block IJK index of each sub-block",
        shape=("*", 3),
        dtype=int,
    )
    subblock_corners = ArrayInstanceProperty(
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

    @property
    def num_cells(self):
        """The number of cells, which in this case are sub-blocks."""
        return None if self.subblock_corners is None else len(self.subblock_corners)

    @property
    def num_parent_blocks(self):
        """The number of parent blocks."""
        return np.prod(self.definition.block_count)

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        return self.num_parent_blocks if location == "parent_blocks" else self.num_cells

    @properties.validator
    def _validate_subblocks(self):
        _shrink_uint(self.subblock_parent_indices)
        _shrink_uint(self.subblock_corners)
        check_subblocks(
            self.definition,
            self.subblock_definition,
            self.subblock_parent_indices.array,
            self.subblock_corners.array,
            instance=self,
        )


class FreeformSubblockedModel(ProjectElement):
    """A regular block model where sub-blocks can be anywhere within the parent."""

    schema = "org.omf.v2.elements.blockmodel.freeform_subblocked"
    _valid_locations = ("cells", "parent_blocks")

    definition = properties.Instance(
        "Block model definition, for the parent blocks",
        RegularBlockModelDefinition,
        default=RegularBlockModelDefinition,
    )
    subblock_definition = properties.Union(
        "Defines the structure of sub-blocks within each parent block.",
        props=[
            properties.Instance("", FreeformSubblockDefinition),
            properties.Instance("", VariableHeightSubblockDefinition),
        ],
        default=FreeformSubblockDefinition,
    )
    subblock_parent_indices = ArrayInstanceProperty(
        "The parent block IJK index of each sub-block",
        shape=("*", 3),
        dtype=int,
    )
    subblock_corners = ArrayInstanceProperty(
        """The positions of the sub-block corners on the grid within their parent block.

        The columns are (min_i, min_j, min_k, max_i, max_j, max_k). Values must be
        between 0.0 and 1.0 inclusive.

        Sub-blocks must stay within the parent block and should not overlap. Gaps are
        allowed but it will be impossible for 'cell' attributes to assign values to
        those areas.
        """,
        shape=("*", 6),
        dtype=float,
    )

    @property
    def num_cells(self):
        """The number of cells, which in this case are always parent blocks."""
        return None if self.subblock_corners is None else len(self.subblock_corners)

    @property
    def num_parent_blocks(self):
        """The number of parent blocks."""
        return np.prod(self.definition.block_count)

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        return self.num_parent_blocks if location == "parent_blocks" else self.num_cells

    @properties.validator
    def _validate_subblocks(self):
        _shrink_uint(self.subblock_parent_indices)
        check_subblocks(
            self.definition,
            self.subblock_definition,
            self.subblock_parent_indices.array,
            self.subblock_corners.array,
            instance=self,
        )
