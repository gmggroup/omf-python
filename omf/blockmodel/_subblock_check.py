import itertools

import numpy as np

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


def _check_parent_indices(definition, parent_indices):
    count = definition.block_count
    if (parent_indices < 0).any() or (parent_indices >= count).any():
        raise IndexError(
            f"0 <= subblock_parent_indices < ({count[0]}, {count[1]}, {count[2]}) failed"
        )


def _check_inside_parent(subblock_definition, corners):
    if isinstance(subblock_definition, RegularSubblockDefinition):
        upper = subblock_definition.count
        upper_str = f"({upper[0]}, {upper[1]}, {upper[2]})"
    else:
        upper = 1.0
        upper_str = "1"
    mn = corners[:, :3]
    mx = corners[:, 3:]
    if mn.dtype.kind != "u" and not (0 <= mn).all():
        raise IndexError("0 <= min_corner failed")
    if not (mn < mx).all():
        raise IndexError("min_corner < max_corner failed")
    if not (mx <= upper).all():
        raise IndexError(f"max_corner <= ({upper_str}) failed")


def check_subblocks(definition, subblock_definition, parent_indices, corners):
    if len(parent_indices) != len(corners):
        raise ValueError(
            "'subblock_parent_indices' and 'subblock_corners' arrays must be the same length"
        )
    _check_inside_parent(subblock_definition, corners)
    _check_parent_indices(definition, parent_indices)
    # Check order and pass groups to the sub-block definition to check further.
    seen = np.zeros(np.prod(definition.block_count), dtype=bool)
    validate = subblock_definition.validate_subblocks
    for start, end, parent in _group_by(definition.ijk_to_index(parent_indices)):
        if seen[parent]:
            raise ValueError(
                "all sub-blocks inside one parent block must be adjacent in the arrays"
            )
        seen[parent] = True
        validate(corners[start:end])
