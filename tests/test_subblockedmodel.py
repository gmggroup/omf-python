"""Tests for block models"""
import numpy as np
import properties
import pytest

import omf
from omf.blockmodel import _subblock_check


def test_group_by():
    """Test the array grouping function used by sub-block checks."""
    group_by = _subblock_check._group_by  # pylint: disable=W0212
    arr = np.array([0, 0, 1, 1, 1, 2])
    assert list(group_by(arr)) == [(0, 2, 0), (2, 5, 1), (5, 6, 2)]
    arr = np.ones(1, dtype=int)
    assert list(group_by(arr)) == [(0, 1, 1)]
    arr = np.zeros(0, dtype=int)
    assert not list(group_by(arr))


def _bm_def():
    return omf.RegularBlockModelDefinition(
        block_size=(1.0, 1.0, 1.0),
        block_count=(1, 1, 1),
    )


def _test_regular(*corners):
    block_model = omf.BlockModel(subblocks=omf.RegularSubblocks(), definition=_bm_def())
    block_model.subblocks.definition = omf.RegularSubblockDefinition(subblock_count=(5, 4, 3))
    block_model.subblocks.corners = np.array(corners)
    block_model.subblocks.parent_indices = np.zeros((len(corners), 3), dtype=int)
    block_model.validate()


def test_overlap():
    """Test that overlapping sub-blocks are rejected."""
    with pytest.raises(properties.ValidationError, match="overlapping sub-blocks"):
        _test_regular((0, 0, 0, 2, 2, 1), (0, 0, 0, 4, 4, 2))


def test_outside_parent():
    """Test that sub-blocks outside the parent block are rejected."""
    with pytest.raises(properties.ValidationError, match="0 <= min_corner"):
        _test_regular((0, 0, -1, 4, 4, 1))
    with pytest.raises(properties.ValidationError, match="min_corner < max_corner"):
        _test_regular((4, 0, 0, 0, 4, 2))
    with pytest.raises(properties.ValidationError, match=r"max_corner <= \(5, 4, 3\)"):
        _test_regular((0, 0, 0, 4, 5, 2))


def test_invalid_parent_indices():
    """Test invalid parent block indices are rejected."""
    block_model = omf.BlockModel(subblocks=omf.RegularSubblocks())
    block_model.definition = _bm_def()
    block_model.subblocks.definition = omf.RegularSubblockDefinition(subblock_count=(5, 4, 3))
    block_model.subblocks.corners = np.array([(0, 0, 0, 5, 4, 3), (0, 0, 0, 5, 4, 3)])
    block_model.subblocks.parent_indices = np.array([(0, 0, 0), (1, 0, 0)])
    with pytest.raises(properties.ValidationError, match=r"subblock_parent_indices < \(1, 1, 1\)"):
        block_model.validate()
    block_model.subblocks.parent_indices = np.array([(0, 0, -1), (0, 0, 0)])
    with pytest.raises(properties.ValidationError, match="0 <= subblock_parent_indices"):
        block_model.validate()


def _test_octree(*corners):
    block_model = omf.BlockModel(
        definition=_bm_def(),
        subblocks=omf.RegularSubblocks(definition=omf.OctreeSubblockDefinition(subblock_count=(4, 4, 2))),
    )
    block_model.subblocks.corners = np.array(corners)
    block_model.subblocks.parent_indices = np.zeros((len(corners), 3), dtype=int)
    block_model.validate()


def test_one_full_block():
    """Test a single sub-block covering the parent."""
    _test_octree((0, 0, 0, 4, 4, 2))


def test_eight_blocks():
    """Test eight sub-blocks covering the parent."""
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
    """Test that non-octree sub-blocks sizes are rejected."""
    with pytest.raises(properties.ValidationError, match="non-octree sub-block sizes"):
        _test_octree((0, 0, 0, 3, 4, 2))


def test_bad_position():
    """Test that non-octree sub-blocks positions are rejected."""
    with pytest.raises(properties.ValidationError, match="non-octree sub-block positions"):
        _test_octree((0, 1, 0, 2, 3, 1))


def test_pack_subblock_arrays():
    """Test that packing of uint arrays during validation works."""
    block_model = omf.BlockModel()
    block_model.subblocks = omf.RegularSubblocks()
    block_model.subblocks.definition.subblock_count = [2, 2, 2]
    block_model.definition.block_size = [1.0, 1.0, 1.0]
    block_model.definition.block_count = [10, 10, 10]
    block_model.subblocks.parent_indices = np.array([(0, 0, 0)], dtype=int)
    block_model.subblocks.corners = np.array([(0, 0, 0, 2, 2, 2)], dtype=int)
    block_model.validate()
    # Arrays were set as int, validate should have packed it down to uint8.
    assert block_model.subblocks.corners.array.dtype == np.uint8


def test_uninstantiated():
    """Test that definitions are default and attributes are None on instantiation"""
    block_model = omf.BlockModel(subblocks=omf.RegularSubblocks())
    assert isinstance(block_model.definition, omf.RegularBlockModelDefinition)
    assert isinstance(block_model.subblocks.definition, omf.RegularSubblockDefinition)
    assert block_model.definition.block_count is None
    assert block_model.subblocks.definition.subblock_count is None
    assert block_model.num_cells is None
    assert block_model.subblocks.parent_indices is None
    assert block_model.subblocks.corners is None
    np.testing.assert_array_equal(block_model.definition.block_size, (1.0, 1.0, 1.0))


def test_num_cells():
    """Test num_cells calculation is correct"""
    block_model = omf.BlockModel(subblocks=omf.RegularSubblocks())
    block_model.definition.block_count = [2, 2, 2]
    block_model.definition.block_size = [1.0, 2.0, 3.0]
    block_model.subblocks.definition.subblock_count = [5, 5, 5]
    np.testing.assert_array_equal(block_model.definition.block_count, [2, 2, 2])
    np.testing.assert_array_equal(block_model.subblocks.definition.subblock_count, [5, 5, 5])
    block_model.subblocks.parent_indices = np.array([(0, 0, 0), (1, 0, 0)])
    block_model.subblocks.corners = np.array([(0, 0, 0, 5, 5, 5), (1, 1, 1, 4, 4, 4)])
    assert block_model.num_cells == 2
    assert block_model.num_parent_blocks == 8
    assert block_model.location_length("cells") == 2
    assert block_model.location_length("parent_blocks") == 8
