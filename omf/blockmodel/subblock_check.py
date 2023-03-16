"""blockmodel/_subblock_check.py: functions for checking sub-block constraints."""
from dataclasses import dataclass

import numpy as np
import properties

from .index import ijk_to_index
from .subblocks import FreeformSubblocks, RegularSubblocks

__all__ = ["subblock_check"]


def _group_by(arr):
    if len(arr) == 0:
        return
    diff = np.flatnonzero(arr[1:] != arr[:-1])
    diff += 1
    if len(diff) == 0:
        yield 0, len(arr), arr[0]
    else:
        yield 0, diff[0], arr[0]
        for start, end in zip(diff[:-1], diff[1:]):
            yield start, end, arr[start]
        yield diff[-1], len(arr), arr[-1]


def _sizes_to_ints(sizes):
    sizes = np.array(sizes, dtype=np.uint64)
    assert len(sizes.shape) == 2 and sizes.shape[1] == 3
    sizes[:, 0] *= 2**32
    sizes[:, 1] *= 2**16
    return sizes.sum(axis=1)


@dataclass
class _Checker:
    # pylint: disable=too-many-instance-attributes
    parent_indices: np.ndarray
    corners: np.ndarray
    block_count: np.ndarray
    subblock_count: np.ndarray
    regular: bool = False
    octree: bool = False
    full: bool = False
    instance: object = None

    def check(self):
        """Run all checks on the given defintions and sub-blocks."""
        if len(self.parent_indices) != len(self.corners):
            self._error(
                "'subblock_parent_indices' and 'subblock_corners' arrays must be the same length",
                prop="subblock_corners",
            )
        self._check_inside_parent()
        self._check_parent_indices()
        if self.regular:
            if self.octree:
                self._check_octree()
            elif self.full:
                self._check_full()
            self._check_for_overlaps()

    def _error(self, message, prop=None):
        raise properties.ValidationError(message, prop=prop, instance=self.instance)

    def _check_parent_indices(self):
        if (self.parent_indices < 0).any() or (self.parent_indices >= self.block_count).any():
            self._error(
                f"0 <= subblock_parent_indices < {self.block_count} failed",
                prop="subblock_parent_indices",
            )

    def _check_inside_parent(self):
        min_corners = self.corners[:, :3]
        max_corners = self.corners[:, 3:]
        if min_corners.dtype.kind != "u" and not (0 <= min_corners).all():
            self._error("0 <= min_corner failed", prop="subblock_corners")
        if not (min_corners < max_corners).all():
            self._error("min_corner < max_corner failed", prop="subblock_corners")
        upper = 1.0 if self.subblock_count is None else self.subblock_count
        if not (max_corners <= upper).all():
            self._error(f"max_corner <= {upper} failed", prop="subblock_corners")

    def _check_octree(self):
        min_corners = self.corners[:, :3]
        max_corners = self.corners[:, 3:]
        sizes = max_corners - min_corners
        # Sizes.
        count = self.subblock_count.copy()
        valid_sizes = [count.copy()]
        while (count > 1).any():
            count[count > 1] //= 2
            valid_sizes.append(count.copy())
        valid_sizes = _sizes_to_ints(valid_sizes)
        if not np.isin(_sizes_to_ints(sizes), valid_sizes).all():
            self._error("found non-octree sub-block sizes", prop="subblock_corners")
        # Positions; octree blocks always start at a multiple of their size.
        if (np.remainder(min_corners, sizes) != 0).any():
            self._error("found non-octree sub-block positions", prop="subblock_corners")

    def _check_full(self):
        valid_sizes = _sizes_to_ints([self.subblock_count, (1, 1, 1)])
        sizes = self.corners[:, 3:] - self.corners[:, :3]
        if not np.isin(_sizes_to_ints(sizes), valid_sizes).all():
            self._error("found sub-block size that does not match 'full' mode'", prop="subblock_corners")

    def _check_for_overlaps(self):
        seen = np.zeros(np.prod(self.block_count), dtype=bool)
        for start, end, value in _group_by(ijk_to_index(self.block_count, self.parent_indices)):
            if seen[value]:
                self._error(
                    "all sub-blocks inside one parent block must be adjacent in the arrays",
                    prop="subblock_parent_indices",
                )
            seen[value] = True
            if end - start > 1:
                self._check_group_for_overlaps(self.corners[start:end])

    def _check_group_for_overlaps(self, corners_in_one_parent):
        # This won't be very fast but there doesn't seem to be a better option.
        tracker = np.zeros(self.subblock_count[::-1], dtype=int)
        for min_i, min_j, min_k, max_i, max_j, max_k in corners_in_one_parent:
            tracker[min_k:max_k, min_j:max_j, min_i:max_i] += 1
        if (tracker > 1).any():
            self._error("found overlapping sub-blocks", prop="subblock_corners")


def subblock_check(model):
    """Checks the sub-blocks in the given block model, if any.

    Raises properties.ValidationError if there is a problem.
    """
    if isinstance(model.subblocks, RegularSubblocks):
        checker = _Checker(
            parent_indices=model.subblocks.parent_indices.array,
            corners=model.subblocks.corners.array,
            block_count=model.block_count,
            subblock_count=model.subblocks.subblock_count,
            regular=True,
            octree=model.subblocks.mode == "octree",
            full=model.subblocks.mode == "full",
            instance=model.subblocks,
        )
    elif isinstance(model.subblocks, FreeformSubblocks):
        checker = _Checker(
            parent_indices=model.subblocks.parent_indices.array,
            corners=model.subblocks.corners.array,
            block_count=model.block_count,
            subblock_count=np.array((1.0, 1.0, 1.0)),
            instance=model.subblocks,
        )
    else:
        return
    checker.check()
