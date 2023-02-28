import numpy as np
import properties

from ..base import ProjectElement
from .definition import RegularBlockModelDefinition, TensorBlockModelDefinition
from .subblocks import FreeformSubblockDefinition, RegularSubblockDefinition
from ._subblock_check import check_subblocks


def _shrink_uint(arr):
    assert arr.min() >= 0
    t = np.min_scalar_type(arr.max())
    return arr.astype(t)


class RegularBlockModel(ProjectElement):
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
        match location:
            case "cells" | "parent_blocks" | "":
                return self.num_cells
            case _:
                raise ValueError(f"unknown location type: {location!r}")


class TensorGridBlockModel(ProjectElement):
    schema = "org.omf.v2.elements.blockmodel.tensor"
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
        bc = self.definition.block_count
        return None if bc is None else np.prod(bc + 1)

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        match location:
            case "cells" | "parent_blocks" | "":
                return self.num_cells
            case "vertices":
                return self.num_nodes
            case _:
                raise ValueError(f"unknown location type: {location!r}")


class SubblockedModel(ProjectElement):
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
        """The number of cells, which in this case are always parent blocks."""
        return None if self.subblock_corners is None else len(self.subblock_corners)

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        match location:
            case "cells" | "":
                return self.num_cells
            case "parent_blocks":
                return np.prod(self.definition.block_count)
            case _:
                raise ValueError(f"unknown location type: {location!r}")

    @properties.validator
    def _validate_subblocks(self):
        check_subblocks(
            self.definition,
            self.subblock_definition,
            self.subblock_parent_indices,
            self.subblock_corners,
            instance=self,
        )
        self.subblock_parent_indices = _shrink_uint(self.subblock_parent_indices)
        self.subblock_corners = _shrink_uint(self.subblock_corners)


class FreeformSubblockedModel(ProjectElement):
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

    def location_length(self, location):
        """Return correct attribute length for 'location'."""
        match location:
            case "cells" | "":
                return self.num_cells
            case "parent_blocks":
                return np.prod(self.definition.block_count)
            case _:
                raise ValueError(f"unknown location type: {location!r}")

    @properties.validator
    def _validate_subblocks(self):
        check_subblocks(
            self.definition,
            self.subblock_definition,
            self.subblock_parent_indices,
            self.subblock_corners,
            instance=self,
        )
        self.subblock_parent_indices = _shrink_uint(self.subblock_parent_indices)
