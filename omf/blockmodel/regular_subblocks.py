"""blockmodel/subblocks.py: sub-block definitions and containers."""
import numpy as np
import properties

from ..attribute import ArrayInstanceProperty
from ..base import BaseModel
from ._subblock_check import check_subblocks, shrink_uint

__all__ = ["OctreeSubblockDefinition", "RegularSubblockDefinition", "RegularSubblocks"]


class RegularSubblockDefinition(BaseModel):
    """The simplest gridded sub-block definition.

    Divide the parent block into a regular grid of `subblock_count` cells. Each block covers
    a cuboid region within that grid. If a parent block is not sub-blocked then it will still
    contain a single block that covers the entire grid.
    """

    schema = "org.omf.v2.blockmodel.subblocks.definition.regular"

    subblock_count = properties.Array(
        "The maximum number of sub-blocks inside a parent in each direction.", dtype=int, shape=(3,)
    )

    @properties.validator("subblock_count")
    def _validate_subblock_count(self, change):
        for item in change["value"]:
            if item < 1:
                raise properties.ValidationError("sub-block counts must be >= 1", prop=change["name"], instance=self)


class OctreeSubblockDefinition(BaseModel):
    """Sub-blocks form an octree inside the parent block.

    Cut the parent block in half in all directions to create eight sub-blocks. Repeat that
    division for some or all of those new sub-blocks. Continue doing that until the limit
    on sub-block count is reached or until the sub-blocks accurately model the inputs.

    This definition also allows the lower level cuts to be omitted in one or two axes,
    giving a maximum sub-block count of (16, 16, 4) for example rather than requiring
    all axes to be equal.
    """

    schema = "org.omf.v2.blockmodel.subblocks.definition.octree"

    subblock_count = properties.Array(
        "The maximum number of sub-blocks inside a parent in each direction.", dtype=int, shape=(3,)
    )

    @properties.validator("subblock_count")
    def _validate_subblock_count(self, change):
        for item in change["value"]:
            if item < 1:
                raise properties.ValidationError("sub-block counts must be >= 1", prop=change["name"], instance=self)
            log = np.log2(item)
            if np.trunc(log) != log:
                raise properties.ValidationError(
                    "octree sub-block counts must be powers of two", prop=change["name"], instance=self
                )


class RegularSubblocks(BaseModel):
    """Defines regular or octree sub-blocks for a block model.

    These sub-blocks must align with a lower-level grid inside the parent block.
    """

    schema = "org.omf.v2.blockmodel.subblocks.regular"

    definition = properties.Union(
        "Defines the structure of sub-blocks within each parent block.",
        props=[RegularSubblockDefinition, OctreeSubblockDefinition],
        default=RegularSubblockDefinition,
    )
    parent_indices = ArrayInstanceProperty(
        "The parent block IJK index of each sub-block",
        shape=("*", 3),
        dtype=int,
    )
    corners = ArrayInstanceProperty(
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

    def validate_subblocks(self, definition):
        """Checks the sub-block data against the given block model definition."""
        shrink_uint(self.parent_indices)
        shrink_uint(self.corners)
        check_subblocks(
            definition, self, instance=self, regular=True, octree=isinstance(self.definition, OctreeSubblockDefinition)
        )

    @property
    def num_subblocks(self):
        """The total number of sub-blocks."""
        return None if self.corners is None else len(self.corners)
