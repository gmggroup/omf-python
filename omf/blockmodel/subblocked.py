"""blockmodel2.py: New Block Model element definitions"""
import numpy as np
import properties

from ._properties import BlockCount, BlockSize
from .regular import BaseBlockModel
from .subblock_definition import FreeformSubblockDefinition, RegularSubblockDefinition


def _shrink_uint(obj, attr):
    arr = getattr(obj, attr)
    assert arr.min() >= 0
    t = np.min_scalar_type(arr.max())
    setattr(obj, attr, arr.astype(t))


class SubblockedModel(BaseBlockModel):
    """A model where regular parent blocks are divided into sub-blocks that align with a grid.

    The sub-blocks here must align with a regular grid within the parent block. What that grid
    is and how blocks are generated within it is defined by the sub-block definition.
    """

    schema = "org.omf.v2.elements.blockmodel.subblocked"

    parent_block_count = BlockCount("Number of parent blocks along u, v, and w axes")
    parent_block_size = BlockSize("Size of parent blocks in the u, v, and w directions")
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
        "Defines the structure of sub-blocks within each parent block for this model.",
        RegularSubblockDefinition,
    )

    @property
    def subblock_count(self):
        return self.subblock_definition.count

    # XXX add num_cells property and implement location_length

    def _check_lengths(self):
        if len(self.subblock_parent_indices) != len(self.subblock_corners):
            raise ValueError(
                "'subblock_parent_indices' and 'subblock_corners' arrays must be the same length"
            )

    def _check_parent_indices(self):
        indices = self.subblock_parent_indices
        count = self.parent_block_count
        if (indices < 0).any() or (indices >= count).any():
            raise IndexError(
                f"0 <= subblock_parent_indices < ({count[0]}, {count[1]}, {count[2]}) failed"
            )

    def _check_inside_parent(self):
        min_corner = self.subblock_corners[:, :3]
        max_corner = self.subblock_corners[:, 3:]
        count = self.subblock_count
        if min_corner.dtype.kind != "u" and not (0 <= min_corner).all():
            raise IndexError("0 <= min_corner failed")
        if not (min_corner < max_corner).all():
            raise IndexError("min_corner < max_corner failed")
        if not (max_corner <= count).all():
            raise IndexError(
                f"max_corner <= ({count[0]}, {count[1]}, {count[2]}) failed"
            )

    @properties.validator
    def _validate_subblocks(self):
        self._check_lengths()
        self._check_parent_indices()
        self._check_inside_parent()
        # Check corners against the definition.
        # TODO check that sub-blocks in each parent are adjacent or remove that requirement
        indices = self.ijk_to_index(self.subblock_parent_indices)
        for index in np.unique(indices):
            corners = self.subblock_corners[indices == index, :]  # XXX slow
            self.subblock_definition.validate_subblocks(corners)
        # Cast to the smallest unsigned integer type.
        _shrink_uint(self, "subblock_parent_indices")
        _shrink_uint(self, "subblock_corners")


class FreeformSubblockedModel(BaseBlockModel):
    """A model where regular parent blocks are divided into free-form sub-blocks."""

    schema = "org.omf.v2.elements.blockmodel.freeform_subblocked"

    parent_block_count = BlockCount("Number of parent blocks along u, v, and w axes")
    parent_block_size = BlockSize("Size of parent blocks in the u, v, and w directions")
    subblock_parent_indices = properties.Array(
        "The parent block IJK index of each sub-block",
        shape=("*", 3),
        dtype=int,
    )
    subblock_corners = properties.Array(
        """The positions of the sub-block corners within their parent block.

        Positions are relative to the parent block, with 0.0 being the minimum side and 1.0 the
        maximum side.

        Sub-blocks must stay within the parent block and should not overlap. Gaps are allowed
        but it will be impossible for 'cell' attributes to assign values to those areas.
        """,
        shape=("*", 6),
        dtype=float,
    )
    subblock_definition = properties.Instance(
        "Defines the structure of sub-blocks within each parent block for this model.",
        FreeformSubblockDefinition,
    )
