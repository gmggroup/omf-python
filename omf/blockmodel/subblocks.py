import numpy as np
import properties

from ._properties import BlockCount, OctreeSubblockCount


class RegularSubblockDefinition(properties.HasProperties):
    """The simplest gridded sub-block definition."""

    subblock_count = BlockCount(
        "The maximum number of sub-blocks inside a parent in each direction."
    )

    @properties.validator("subblock_count")
    def _validate_subblock_count(self, change):
        if (change["value"] > 65535).any():
            raise properties.ValidationError(
                "sub-block count is limited to 65535 in each direction"
            )


class OctreeSubblockDefinition(RegularSubblockDefinition):
    """Sub-blocks form an octree inside the parent block.

    Cut the parent block in half in all directions to create eight sub-blocks. Repeat that
    division for some or all of those new sub-blocks. Continue doing that until the limit
    on sub-block count is reached or until the sub-blocks accurately model the inputs.

    This definition also allows the lower level cuts to be omitted in one or two axes,
    giving a maximum sub-block count of (16, 16, 4) for example rather than requiring
    all axes to be equal.
    """

    subblock_count = OctreeSubblockCount(
        "The maximum number of sub-blocks inside a parent in each direction."
    )


class FreeformSubblockDefinition:
    """Unconstrained free-form sub-block definition.

    Provide np limitations on, or explanation of, sub-block positions.
    """


class VariableZSubblockDefinition(FreeformSubblockDefinition):
    """Sub-blocks will be contrained to be on an XY grid with variable Z."""

    # FIXME add var-z properties
