"""Tests for block models"""
import numpy as np
import properties
import pytest

import omf
from omf.blockmodel import _subblock_check


def test_group_by():
    a = np.array([0, 0, 1, 1, 1, 2])
    assert list(_subblock_check._group_by(a)) == [(0, 2, 0), (2, 5, 1), (5, 6, 2)]
    a = np.ones(1, dtype=int)
    assert list(_subblock_check._group_by(a)) == [(0, 1, 1)]
    a = np.zeros(0, dtype=int)
    assert list(_subblock_check._group_by(a)) == []


def _bm_def():
    return omf.RegularBlockModelDefinition(
        block_size=(1.0, 1.0, 1.0),
        block_count=(1, 1, 1),
    )


def _test_regular(*corners):
    block_model = omf.SubblockedModel()
    block_model.definition = _bm_def()
    block_model.subblock_definition = omf.RegularSubblockDefinition(
        subblock_count=(5, 4, 3)
    )
    block_model.subblock_corners = np.array(corners)
    block_model.subblock_parent_indices = np.zeros((len(corners), 3), dtype=int)
    block_model.validate()


def test_overlap():
    with pytest.raises(properties.ValidationError, match="overlapping sub-blocks"):
        _test_regular((0, 0, 0, 2, 2, 1), (0, 0, 0, 4, 4, 2))


def test_outside_parent():
    with pytest.raises(properties.ValidationError, match="0 <= min_corner"):
        _test_regular((0, 0, -1, 4, 4, 1))
    with pytest.raises(properties.ValidationError, match="min_corner < max_corner"):
        _test_regular((4, 0, 0, 0, 4, 2))
    with pytest.raises(properties.ValidationError, match=r"max_corner <= \(5, 4, 3\)"):
        _test_regular((0, 0, 0, 4, 5, 2))


def test_invalid_parent_indices():
    block_model = omf.SubblockedModel()
    block_model.definition = _bm_def()
    block_model.subblock_definition = omf.RegularSubblockDefinition(
        subblock_count=(5, 4, 3)
    )
    block_model.subblock_corners = np.array([(0, 0, 0, 5, 4, 3), (0, 0, 0, 5, 4, 3)])
    block_model.subblock_parent_indices = np.array([(0, 0, 0), (1, 0, 0)])
    with pytest.raises(
        properties.ValidationError, match=r"subblock_parent_indices < \(1, 1, 1\)"
    ):
        block_model.validate()
    block_model.subblock_parent_indices = np.array([(0, 0, -1), (0, 0, 0)])
    with pytest.raises(
        properties.ValidationError, match="0 <= subblock_parent_indices"
    ):
        block_model.validate()


def _test_octree(*corners):
    block_model = omf.SubblockedModel()
    block_model.definition = _bm_def()
    block_model.subblock_definition = omf.OctreeSubblockDefinition(
        subblock_count=(4, 4, 2)
    )
    block_model.subblock_corners = np.array(corners)
    block_model.subblock_parent_indices = np.zeros((len(corners), 3), dtype=int)
    block_model.validate()


def test_one_full_block():
    _test_octree((0, 0, 0, 4, 4, 2))


def test_eight_blocks():
    _test_octree(
        (0, 0, 0, 2, 2, 1),
        (2, 0, 0, 4, 2, 1),
        (0, 2, 0, 2, 4, 1),
        (2, 2, 0, 4, 4, 1),
        (0, 0, 1, 2, 2, 2),
        (2, 0, 1, 4, 2, 2),
        (0, 2, 1, 2, 4, 2),
        (2, 2, 1, 4, 4, 2),
    )


def test_bad_size():
    with pytest.raises(properties.ValidationError, match="non-octree sub-block sizes"):
        _test_octree((0, 0, 0, 3, 4, 2))


def test_bad_position():
    with pytest.raises(
        properties.ValidationError, match="non-octree sub-block positions"
    ):
        _test_octree((0, 1, 0, 2, 3, 1))
