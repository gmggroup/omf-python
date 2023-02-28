import itertools

import numpy as np
import properties

from .subblocks import RegularSubblockDefinition


def _group_by(arr):
    assert len(arr) > 0
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
        upper = subblock_definition.count
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
            f"max_corner <= ({upper_str}) failed",
            prop="subblock_corners",
            instance=instance,
        )


def check_subblocks(definition, subblock_definition, parent_indices, corners, instance):
    if len(parent_indices) != len(corners):
        raise properties.ValidationError(
            "'subblock_parent_indices' and 'subblock_corners' arrays must be the same length",
            prop="subblock_corners",
            instance=instance,
        )
    _check_inside_parent(subblock_definition, corners, instance)
    _check_parent_indices(definition, parent_indices, instance)
    # Check order and pass groups to the sub-block definition to check further.
    seen = np.zeros(np.prod(definition.block_count), dtype=bool)
    validate = subblock_definition.validate_subblocks
    for start, end, parent in _group_by(definition.ijk_to_index(parent_indices)):
        if seen[parent]:
            raise properties.ValidationError(
                "all sub-blocks inside one parent block must be adjacent in the arrays",
                prop="subblock_parent_indices",
                instance=instance,
            )
        seen[parent] = True
        validate(corners[start:end])
