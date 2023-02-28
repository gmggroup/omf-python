import properties

from ._properties import BlockCount, OctreeSubblockCount


class RegularSubblockDefinition(properties.HasProperties):
    """The simplest gridded sub-block definition."""

    count = BlockCount(
        "The maximum number of sub-blocks inside a parent in each direction."
    )

    def validate_subblocks(self, _corners):
        """Checks the sub-blocks within one parent block."""
        # TODO check for overlaps


class OctreeSubblockDefinition(RegularSubblockDefinition):
    """Sub-blocks form an octree inside the parent block.

    Cut the parent block in half in all directions to create eight sub-blocks. Repeat that
    division for some or all of those new sub-blocks. Continue doing that until the limit
    on sub-block count is reached or until the sub-blocks accurately model the inputs.

    This definition also allows the lower level cuts to be omitted in one or two axes,
    giving a maximum sub-block count of (16, 16, 4) for example rather than requiring
    all axes to be equal.
    """

    count = OctreeSubblockCount(
        "The maximum number of sub-blocks inside a parent in each direction."
    )

    def validate_subblocks(self, corners):
        """Checks the sub-blocks within one parent block."""
        super().validate_subblocks(corners)
        # TODO check that blocks lie on the octree


class FreeformSubblockDefinition:
    """Unconstrained free-form sub-block definition.

    Provide np limitations on, or explanation of, sub-block positions.
    """

    def validate_subblocks(self, _corners):
        """Checks the sub-blocks within one parent block."""
        # XXX can we check for overlaps efficiently?


class VariableZSubblockDefinition(FreeformSubblockDefinition):
    def validate_subblocks(self, corners):
        """Checks the sub-blocks within one parent block."""
        super().validate_subblocks(corners)
        # TODO check that blocks lie on the octree
