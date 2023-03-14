"""blockmodel/subblocks.py: sub-block definitions and containers."""
import numpy as np
import properties

from ..attribute import ArrayInstanceProperty
from ..base import BaseModel
from ._utils import shrink_uint, SubblockChecker

__all__ = ["RegularSubblocks", "SubblockModeOctree", "SubblockModeFull"]


class SubblockModeFull(BaseModel):
    """The parent is fully sub-blocked, with each sub-block having size (1, 1, 1).

    The importing app may want to merge cells to make this more efficient.
    """

    schema = "org.omf.v2.elements.blockmodel.subblocks.mode_full"


class SubblockModeOctree(BaseModel):
    """Sub-blocks form an octree inside the parent block.

    Cut the parent block in half in all directions to create eight sub-blocks. Repeat that
    division for some or all of those new sub-blocks. Continue doing that until the limit
    on sub-block count is reached or until the sub-blocks accurately model the inputs.

    This definition also allows the lower level cuts to be omitted in one or two axes,
    giving a maximum sub-block count of (16, 16, 4) for example rather than requiring
    all axes to be equal.
    """

    schema = "org.omf.v2.elements.blockmodel.subblocks.mode_octree"


class RegularSubblocks(BaseModel):
    """Defines regular or octree sub-blocks for a block model.

    Divide the parent block into a regular grid of `subblock_count` cells. Each block covers
    a cuboid region within that grid.
    """

    schema = "org.omf.v2.elements.blockmodel.subblocks.regular"

    subblock_count = properties.Array(
        "The maximum number of sub-blocks inside a parent in each direction.", dtype=int, shape=(3,)
    )
    mode = properties.Union(
        "Defines the structure of sub-blocks within each parent block.",
        props=[SubblockModeFull, SubblockModeOctree],
        required=False,
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
        those areas. A paret that is not sub-blocked but does have attributes should
        be represented as a sub-block that covers the entire parent block.
        """,
        shape=("*", 6),
        dtype=int,
    )

    @properties.validator("subblock_count")
    def _validate_subblock_count(self, change):
        for item in change["value"]:
            if item < 1:
                raise properties.ValidationError("sub-block counts must be >= 1", prop=change["name"], instance=self)

    @properties.validator
    def _validate(self):
        if isinstance(self.mode, SubblockModeOctree):
            for item in self.subblock_count:
                log = np.log2(item)
                if np.trunc(log) != log:
                    raise properties.ValidationError(
                        "in octree mode sub-block counts must be powers of two", prop="subblock_count", instance=self
                    )

    def validate_subblocks(self, model):
        """Checks the sub-block data against the given block model definition."""
        shrink_uint(self.parent_indices)
        shrink_uint(self.corners)
        checker = SubblockChecker.from_regular(model)
        if isinstance(self.mode, SubblockModeOctree):
            checker.octree = True
        if isinstance(self.mode, SubblockModeFull):
            checker.full = True
        checker.check()

    @property
    def num_subblocks(self):
        """The total number of sub-blocks."""
        return None if self.corners is None else len(self.corners)
