import itertools
import sys

import numpy as np
import properties

from .subblocks import RegularSubblockDefinition, OctreeSubblockDefinition


def _group_by(arr):
    if len(arr) == 0:
        return
    diff = np.flatnonzero(arr[1:] != arr[:-1])
    diff += 1
    if len(diff) == 0:
        yield 0, len(arr), arr[0]
    else:
        yield 0, diff[0], arr[0]
        for start, end in itertools.pairwise(diff):
            yield start, end, arr[start]
        yield diff[-1], len(arr), arr[-1]


def _check_parent_indices(definition, parent_indices, instance):
    count = definition.block_count
    if (parent_indices < 0).any() or (parent_indices >= count).any():
        raise properties.ValidationError(
            f"0 <= subblock_parent_indices < ({count[0]}, {count[1]}, {count[2]}) failed",
            prop="subblock_parent_indices",
            instance=instance,
        )


def _check_inside_parent(subblock_definition, corners, instance):
    if isinstance(subblock_definition, RegularSubblockDefinition):
        upper = subblock_definition.subblock_count
        upper_str = f"({upper[0]}, {upper[1]}, {upper[2]})"
    else:
        upper = 1.0
        upper_str = "1"
    mn = corners[:, :3]
    mx = corners[:, 3:]
    if mn.dtype.kind != "u" and not (0 <= mn).all():
        raise properties.ValidationError(
            "0 <= min_corner failed", prop="subblock_corners", instance=instance
        )
    if not (mn < mx).all():
        raise properties.ValidationError(
            "min_corner < max_corner failed", prop="subblock_corners", instance=instance
        )
    if not (mx <= upper).all():
        raise properties.ValidationError(
            f"max_corner <= {upper_str} failed",
            prop="subblock_corners",
            instance=instance,
        )


def _check_for_overlaps(subblock_definition, one_parent_corners, instance):
    # This won't be very fast but there doesn't seem to be a better option.
    tracker = np.zeros(subblock_definition.subblock_count[::-1], dtype=int)
    for min_i, min_j, min_k, max_i, max_j, max_k in one_parent_corners:
        tracker[min_k:max_k, min_j:max_j, min_i:max_i] += 1
    if (tracker > 1).any():
        raise properties.ValidationError(
            "found overlapping sub-blocks", prop="subblock_corners", instance=instance
        )


def _sizes_to_ints(sizes):
    sizes = np.array(sizes, dtype=np.uint64)
    assert len(sizes.shape) == 2 and sizes.shape[1] == 3
    sizes[:, 0] *= 2**32
    sizes[:, 1] *= 2**16
    return sizes.sum(axis=1)


def _check_octree(subblock_definition, corners, instance):
    mn = corners[:, :3]
    mx = corners[:, 3:]
    # Sizes.
    count = subblock_definition.subblock_count
    valid_sizes = [count.copy()]
    while (count > 1).any():
        count[count > 1] //= 2
        valid_sizes.append(count.copy())
    valid_sizes = _sizes_to_ints(valid_sizes)
    sizes = _sizes_to_ints(mx - mn)
    if not np.isin(sizes, valid_sizes, kind="table").all():
        raise properties.ValidationError(
            "found non-octree sub-block sizes",
            prop="subblock_corners",
            instance=instance,
        )
    # Positions. Octree blocks always start at a multiple of their size.
    r = np.remainder(mn, mx - mn)
    if (r != 0).any():
        raise properties.ValidationError(
            "found non-octree sub-block positions",
            prop="subblock_corners",
            instance=instance,
        )


def check_subblocks(
    definition, subblock_definition, parent_indices, corners, instance=None
):
    if len(parent_indices) != len(corners):
        raise properties.ValidationError(
            "'subblock_parent_indices' and 'subblock_corners' arrays must be the same length",
            prop="subblock_corners",
            instance=instance,
        )
    _check_inside_parent(subblock_definition, corners, instance)
    _check_parent_indices(definition, parent_indices, instance)
    if isinstance(subblock_definition, OctreeSubblockDefinition):
        _check_octree(subblock_definition, corners, instance)
    seen = np.zeros(np.prod(definition.block_count), dtype=bool)
    for start, end, value in _group_by(definition.ijk_to_index(parent_indices)):
        if seen[value]:
            raise properties.ValidationError(
                "all sub-blocks inside one parent block must be adjacent in the arrays",
                prop="subblock_parent_indices",
                instance=instance,
            )
        seen[value] = True
        if end - start > 1:
            _check_for_overlaps(subblock_definition, corners[start:end], instance)
