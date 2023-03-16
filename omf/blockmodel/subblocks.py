"""blockmodel/subblocks.py: sub-block definitions and containers."""
from enum import Enum

import numpy as np
import properties

from ..attribute import ArrayInstanceProperty
from ..base import BaseModel

__all__ = ["FreeformSubblocks", "RegularSubblocks"]


def _shrink_uint(arr):
    kind = arr.array.dtype.kind
    if kind == "u" or (kind == "i" and arr.array.min() >= 0):
        arr.array = arr.array.astype(np.min_scalar_type(arr.array.max()))


class RegularSubblocks(BaseModel):
    """Defines regular sub-blocks for a block model.

    Divide the parent block into a regular grid of `subblock_count` cells. Each sub-block
    covers a cuboid region within that grid and they must not overlap. Sub-blocks are
    described by the `parent_indices` and `corners` arrays.

    Each row in `parent_indices` is an IJK index on the block model grid. Each row of
    `corners` is (i_min, j_min, k_min, i_max, j_max, k_max) all integers that refer to
    vertices of the sub-block grid within the parent block. For example:

    * A block with minimum size in the corner of the parent block would be (0, 0, 0, 1, 1, 1).
    * If the `subblock_count` is (5, 5, 3) then a sub-block covering the whole parent would
      be (0, 0, 0, 5, 5, 3).

    Sub-blocks must stay within their parent, must have a non-zero size in all directions, and
    must not overlap. Further contraints can be added by setting `mode`:

    "octree" mode
        Sub-blocks form a modified octree inside the parent block. To form this structure,
        cut the parent block in half in all directions to create eight child blocks.
        Repeat that cut for some or all of those children, and continue doing that until the
        limit on sub-block count is reached or until the sub-blocks accurately model the inputs.

        This modified form of an octree also allows for stopping all cuts in one or two directions
        before the others, so that the `subblock_count` can be (8, 4, 2) rather than (8, 8, 8).
        All values in `subblock_count` must be a powers of two but they don't have to be equal.

    "full" mode
        The parent blocks must be either fully sub-blocked or not sub-blocked at all.
        In other words sub-blocks must either cover the whole parent or have size (1, 1, 1).
    """

    schema = "org.omf.v2.elements.blockmodel.subblocks.regular"

    subblock_count = properties.Array(
        "The maximum number of sub-blocks inside a parent in each direction", dtype=int, shape=(3,)
    )
    mode = properties.StringChoice(
        "Extra constraints on the placement of sub-blocks",
        choices=["octree", "full"],
        required=False,
    )
    parent_indices = ArrayInstanceProperty(
        "The sub-block parent IJK indices",
        shape=("*", 3),
        dtype=int,
    )
    corners = ArrayInstanceProperty(
        """The integer positions of the sub-block corners on the grid within their parent block""",
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
        _shrink_uint(self.parent_indices)
        _shrink_uint(self.corners)
        if self.mode == "octree":
            for item in self.subblock_count:
                log = np.log2(item)
                if np.trunc(log) != log:
                    raise properties.ValidationError(
                        "in octree mode sub-block counts must be powers of two", prop="subblock_count", instance=self
                    )

    @property
    def num_subblocks(self):
        """The total number of sub-blocks."""
        return None if self.corners is None else len(self.corners)


class FreeformSubblocks(BaseModel):
    """Defines free-form sub-blocks for a block model.

    Sub-blocks are described by the `parent_indices` and `corners` arrays.

    Each row in `parent_indices` is an IJK index on the block model grid. Each row of
    `corners` is (i_min, j_min, k_min, i_max, j_max, k_max) all floating point, with 0.0
    referring to the minimum side of the parent block and 1.0 the maximum side.
    For example:

    * A block with size (0.3333, 0.1, 0.5) in the corner of the parent block would be
      (0.0, 0.0, 0.0, 0.3333, 0.1, 0.5).
    * A sub-block covering the whole parent would be (0.0, 0.0, 0.0, 1.0, 1.0, 1.0).

    Sub-blocks must stay within their parent and must have a non-zero size in all directions.
    They shouldn't overlap but that isn't checked as it would take too long.
    """

    schema = "org.omf.v2.elements.blockmodel.subblocks.freeform"

    parent_indices = ArrayInstanceProperty(
        "The parent block IJK index of each sub-block",
        shape=("*", 3),
        dtype=int,
    )
    corners = ArrayInstanceProperty(
        """The positions of the sub-block corners on the grid within their parent block""",
        shape=("*", 6),
        dtype=float,
    )

    @properties.validator
    def _validate(self):
        _shrink_uint(self.parent_indices)

    @property
    def num_subblocks(self):
        """The total number of sub-blocks."""
        return None if self.corners is None else len(self.corners)
