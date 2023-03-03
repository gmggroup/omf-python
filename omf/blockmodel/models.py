"""blockmodel/models.py: block model elements."""
import numpy as np
import properties

from ..base import ProjectElement
from .definition import (
    FreeformSubblockDefinition,
    RegularBlockModelDefinition,
    RegularSubblockDefinition,
    TensorBlockModelDefinition,
)
from ._subblock_check import check_subblocks


def _shrink_uint(arr):
    assert arr.dtype.kind in "ui"
    if arr.min() < 0:
        return arr
    return arr.astype(np.min_scalar_type(arr.max()))


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
        "Defines the structure of sub-blocks within each parent block.",
        RegularSubblockDefinition,
        default=RegularSubblockDefinition,
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
        self.subblock_parent_indices = _shrink_uint(self.subblock_parent_indices)
        self.subblock_corners = _shrink_uint(self.subblock_corners)
        check_subblocks(
            self.definition,
            self.subblock_definition,
            self.subblock_parent_indices,
            self.subblock_corners,
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
    subblock_parent_indices = properties.Array(
        "The parent block IJK index of each sub-block",
        shape=("*", 3),
        dtype=int,
    )
    subblock_corners = properties.Array(
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
    subblock_definition = properties.Instance(
        "Defines the structure of sub-blocks within each parent block.",
        FreeformSubblockDefinition,
        default=FreeformSubblockDefinition,
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
        self.subblock_parent_indices = _shrink_uint(self.subblock_parent_indices)
        check_subblocks(
            self.definition,
            self.subblock_definition,
            self.subblock_parent_indices,
            self.subblock_corners,
            instance=self,
        )
