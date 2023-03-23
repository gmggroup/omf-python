"""Tests for block models"""
import numpy as np
import properties
import pytest

from omf.blockmodel import BlockModel, TensorGrid
from omf.blockmodel.index import ijk_to_index, index_to_ijk


def test_ijk_index_errors():
    """Test ijk indexing into parent blocks errors as expected"""

    with pytest.raises(TypeError):
        ijk_to_index([3, 4, 5], "a")
    with pytest.raises(TypeError):
        index_to_ijk([3, 4, 5], "a")
    with pytest.raises(ValueError):
        ijk_to_index([3, 4, 5], [0, 0])
    with pytest.raises(TypeError):
        ijk_to_index([3, 4, 5], [0, 0, 0.5])
    with pytest.raises(TypeError):
        index_to_ijk([3, 4, 5], 0.5)
    with pytest.raises(IndexError):
        ijk_to_index([3, 4, 5], [0, 0, 5])
    with pytest.raises(IndexError):
        index_to_ijk([3, 4, 5], 60)
    with pytest.raises(IndexError):
        ijk_to_index([3, 4, 5], [[0, 0, 5], [0, 0, 3]])
    with pytest.raises(IndexError):
        index_to_ijk([3, 4, 5], [0, 1, 60])


def test_ijk_index_arrays():
    """Test ijk array indexing into parent blocks works as expected"""
    ijk = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 3, 4)]
    index = [0, 1, 3, 12, 59]
    assert np.array_equal(ijk_to_index([3, 4, 5], ijk), index)
    assert np.array_equal(index_to_ijk([3, 4, 5], index), ijk)
    ijk = [[(0, 0, 0), (1, 0, 0)], [(0, 1, 0), (0, 0, 1)]]
    index = [(0, 1), (3, 12)]
    assert np.array_equal(ijk_to_index([3, 4, 5], ijk), index)
    assert np.array_equal(index_to_ijk([3, 4, 5], index), ijk)


@pytest.mark.parametrize(
    ("ijk", "index"),
    [([0, 0, 0], 0), ([1, 0, 0], 1), ([0, 1, 0], 3), ([0, 0, 1], 12), ([2, 3, 4], 59)],
)
def test_ijk_index(ijk, index):
    """Test ijk indexing into parent blocks works as expected"""
    assert ijk_to_index([3, 4, 5], ijk) == index
    assert np.array_equal(index_to_ijk([3, 4, 5], index), ijk)


def test_tensorblockmodel():
    """Test tensor grid block models."""
    elem = BlockModel(grid=TensorGrid())
    assert elem.num_parent_vertices is None
    assert elem.num_parent_blocks is None
    assert elem.block_count is None
    elem.grid.tensor_u = [1.0, 1.0]
    elem.grid.tensor_v = [2.0, 2.0, 2.0]
    elem.grid.tensor_w = [3.0]
    np.testing.assert_array_equal(elem.block_count, [2, 3, 1])
    assert elem.validate()
    assert elem.location_length("vertices") == 24
    assert elem.location_length("cells") == 6
    elem.axis_v = [1.0, 1.0, 0]
    with pytest.raises(ValueError):
        elem.validate()
    elem.axis_v = "Y"


def test_invalid_tensors():
    """Test invalid tensor arrays on tensor grid block models."""
    elem = BlockModel(grid=TensorGrid())
    with pytest.raises(properties.ValidationError):
        elem.grid.tensor_u = []
    with pytest.raises(properties.ValidationError):
        elem.grid.tensor_v = [1.0, 0.0, 3.0]
    with pytest.raises(properties.ValidationError):
        elem.grid.tensor_w = [-1.0, 2.0]


@pytest.mark.parametrize("block_count", ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5]))
def test_bad_block_count(block_count):
    """Test mismatched block_count"""
    block_model = BlockModel()
    block_model.grid.block_size = [1.0, 2.0, 3.0]
    with pytest.raises((ValueError, properties.ValidationError)):
        block_model.grid.block_count = block_count
        block_model.validate()


@pytest.mark.parametrize("block_size", ([2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [-1.0, 2, 2], [0.0, 2, 2]))
def test_bad_block_size(block_size):
    """Test mismatched block_size"""
    block_model = BlockModel()
    block_model.grid.block_count = [2, 2, 2]
    with pytest.raises((ValueError, properties.ValidationError)):
        block_model.grid.block_size = block_size
        block_model.validate()


def test_uninstantiated():
    """Test all attributes are None on instantiation"""
    block_model = BlockModel()
    assert block_model.grid.block_count is None
    assert block_model.num_parent_blocks is None
    assert block_model.num_parent_vertices is None
    assert block_model.subblocks is None
    np.testing.assert_array_equal(block_model.grid.block_size, (1.0, 1.0, 1.0))


def test_num_cells():
    """Test num_cells calculation is correct"""
    block_model = BlockModel()
    block_model.grid.block_count = [2, 2, 2]
    block_model.grid.block_size = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(block_model.grid.block_count, [2, 2, 2])
    assert block_model.location_length("cells") == 8
    assert block_model.location_length("vertices") == 27
    assert block_model.location_length("parent_blocks") == 8
