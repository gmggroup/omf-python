"""blockmodel/_subblock_check.py: functions for checking sub-block constraints."""
import numpy as np
import properties

from .definition import RegularSubblockDefinition, OctreeSubblockDefinition


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


def _check_parent_indices(definition, parent_indices, instance):
    count = definition.block_count
    if (parent_indices < 0).any() or (parent_indices >= count).any():
        raise properties.ValidationError(
            f"0 <= subblock_parent_indices < ({count[0]}, {count[1]}, {count[2]}) failed",
            prop="subblock_parent_indices",
            instance=instance,
        )


def _check_inside_parent(subblock_definition, corners, instance):
    if subblock_definition.regular:
        upper = subblock_definition.subblock_count
        upper_str = f"({upper[0]}, {upper[1]}, {upper[2]})"
    else:
        upper = 1.0
        upper_str = "1"
    min_corners = corners[:, :3]
    max_corners = corners[:, 3:]
    if min_corners.dtype.kind != "u" and not (0 <= min_corners).all():
        raise properties.ValidationError("0 <= min_corner failed", prop="subblock_corners", instance=instance)
    if not (min_corners < max_corners).all():
        raise properties.ValidationError("min_corner < max_corner failed", prop="subblock_corners", instance=instance)
    if not (max_corners <= upper).all():
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
        raise properties.ValidationError("found overlapping sub-blocks", prop="subblock_corners", instance=instance)


def _sizes_to_ints(sizes):
    sizes = np.array(sizes, dtype=np.uint64)
    assert len(sizes.shape) == 2 and sizes.shape[1] == 3
    sizes[:, 0] *= 2**32
    sizes[:, 1] *= 2**16
    return sizes.sum(axis=1)


def _check_octree(subblock_definition, corners, instance):
    min_corners = corners[:, :3]
    max_corners = corners[:, 3:]
    sizes = max_corners - min_corners
    # Sizes.
    count = subblock_definition.subblock_count
    valid_sizes = [count.copy()]
    while (count > 1).any():
        count[count > 1] //= 2
        valid_sizes.append(count.copy())
    valid_sizes = _sizes_to_ints(valid_sizes)
    if not np.isin(_sizes_to_ints(sizes), valid_sizes).all():
        raise properties.ValidationError(
            "found non-octree sub-block sizes",
            prop="subblock_corners",
            instance=instance,
        )
    # Positions. Octree blocks always start at a multiple of their size.
    if (np.remainder(min_corners, sizes) != 0).any():
        raise properties.ValidationError(
            "found non-octree sub-block positions",
            prop="subblock_corners",
            instance=instance,
        )


def check_subblocks(definition, subblock_definition, parent_indices, corners, instance=None):
    """Run all checks on the given defintions and sub-blocks."""
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
