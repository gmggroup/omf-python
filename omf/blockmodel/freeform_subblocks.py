"""blockmodel/subblocks.py: sub-block definitions and containers."""
import properties

from ..attribute import ArrayInstanceProperty
from ..base import BaseModel
from ._subblock_check import check_subblocks, shrink_uint

__all__ = ["FreeformSubblockDefinition", "FreeformSubblocks", "VariableHeightSubblockDefinition"]


class FreeformSubblockDefinition(BaseModel):
    """Unconstrained free-form sub-block definition.

    Provides no limitations on or explanation of sub-block positions.
    """

    schema = "org.omf.v2.blockmodel.subblocks.definition.freeform"


class VariableHeightSubblockDefinition(BaseModel):
    """Defines sub-blocks on a grid in the U and V directions but variable in the W direction.

    A single sub-block covering the whole parent block is also valid. Sub-blocks should not
    overlap.

    Note: these constraints on sub-blocks are not checked during validation.
    """

    schema = "org.omf.v2.blockmodel.subblocks.definition.varheight"

    subblock_count_u = properties.Integer("Number of sub-blocks in the u-direction", min=1, max=65535)
    subblock_count_v = properties.Integer("Number of sub-blocks in the v-direction", min=1, max=65535)
    minimum_size_w = properties.Float("Minimum size of sub-blocks in the z-direction", min=0.0)


class FreeformSubblocks(BaseModel):
    """Defines free-form sub-blocks for a block model.

    These sub-blocks can exist anywhere without the parent block, subject to any extra
    conditions the sub-block definition imposes.
    """

    schema = "org.omf.v2.blockmodel.subblocks.freeform"

    definition = properties.Union(
        "Defines the structure of sub-blocks within each parent block.",
        props=[FreeformSubblockDefinition, VariableHeightSubblockDefinition],
        default=FreeformSubblockDefinition,
    )
    parent_indices = ArrayInstanceProperty(
        "The parent block IJK index of each sub-block",
        shape=("*", 3),
        dtype=int,
    )
    corners = ArrayInstanceProperty(
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

    def validate_subblocks(self, definition):
        """Checks the sub-block data against the given block model definition."""
        shrink_uint(self.parent_indices)
        shrink_uint(self.corners)
        check_subblocks(definition, self, instance=self)

    @property
    def num_subblocks(self):
        """The total number of sub-blocks."""
        return None if self.corners is None else len(self.corners)
