"""blockmodel/subblocks.py: sub-block definitions and containers."""
import properties

from ..attribute import ArrayInstanceProperty
from ..base import BaseModel
from ._utils import shrink_uint, SubblockChecker

__all__ = ["FreeformSubblocks"]


class FreeformSubblocks(BaseModel):
    """Defines free-form sub-blocks for a block model.

    These sub-blocks can exist anywhere without the parent block, subject to any extra
    conditions the sub-block definition imposes.
    """

    schema = "org.omf.v2.elements.blockmodel.subblocks.freeform"

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

    def validate_subblocks(self, model):
        """Checks the sub-block data against the given block model definition."""
        shrink_uint(self.parent_indices)
        SubblockChecker.from_freeform(model).check()

    @property
    def num_subblocks(self):
        """The total number of sub-blocks."""
        return None if self.corners is None else len(self.corners)
