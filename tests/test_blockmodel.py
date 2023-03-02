"""Tests for block models"""
import numpy as np
import properties
import pytest

import omf


def _make_regular_definition(count):
    return omf.RegularBlockModelDefinition(block_count=count, block_size=(1.0, 1.0, 1.0))


def test_ijk_index_errors():
    """Test ijk indexing into parent blocks errors as expected"""

    defn = _make_regular_definition([3, 4, 5])
    with pytest.raises(TypeError):
        defn.ijk_to_index("a")
    with pytest.raises(TypeError):
        defn.index_to_ijk("a")
    with pytest.raises(ValueError):
        defn.ijk_to_index([0, 0])
    with pytest.raises(TypeError):
        defn.ijk_to_index([0, 0, 0.5])
    with pytest.raises(TypeError):
        defn.index_to_ijk(0.5)
    with pytest.raises(IndexError):
        defn.ijk_to_index([0, 0, 5])
    with pytest.raises(IndexError):
        defn.index_to_ijk(60)
    with pytest.raises(IndexError):
        defn.ijk_to_index([[0, 0, 5], [0, 0, 3]])
    with pytest.raises(IndexError):
        defn.index_to_ijk([0, 1, 60])


def test_ijk_index_arrays():
    """Test ijk array indexing into parent blocks works as expected"""
    defn = _make_regular_definition([3, 4, 5])
    ijk = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 3, 4)]
    index = [0, 1, 3, 12, 59]
    assert np.array_equal(defn.ijk_to_index(ijk), index)
    assert np.array_equal(defn.index_to_ijk(index), ijk)
    ijk = [[(0, 0, 0), (1, 0, 0)], [(0, 1, 0), (0, 0, 1)]]
    index = [(0, 1), (3, 12)]
    assert np.array_equal(defn.ijk_to_index(ijk), index)
    assert np.array_equal(defn.index_to_ijk(index), ijk)


@pytest.mark.parametrize(
    ("ijk", "index"),
    [([0, 0, 0], 0), ([1, 0, 0], 1), ([0, 1, 0], 3), ([0, 0, 1], 12), ([2, 3, 4], 59)],
)
def test_ijk_index(ijk, index):
    """Test ijk indexing into parent blocks works as expected"""
    defn = _make_regular_definition([3, 4, 5])
    assert defn.ijk_to_index(ijk) == index
    assert np.array_equal(defn.index_to_ijk(index), ijk)


def test_tensorblockmodel():
    """Test volume grid geometry validation"""
    elem = omf.TensorGridBlockModel()
    assert elem.num_nodes is None
    assert elem.num_cells is None
    assert elem.definition.block_count is None
    elem.definition.tensor_u = [1.0, 1.0]
    elem.definition.tensor_v = [2.0, 2.0, 2.0]
    elem.definition.tensor_w = [3.0]
    np.testing.assert_array_equal(elem.definition.block_count, [2, 3, 1])
    assert elem.validate()
    assert elem.location_length("vertices") == 24
    assert elem.location_length("cells") == 6
    elem.definition.axis_v = [1.0, 1.0, 0]
    with pytest.raises(ValueError):
        elem.validate()
    elem.axis_v = "Y"


@pytest.mark.parametrize("block_count", ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5]))
def test_bad_block_count(block_count):
    """Test mismatched block_count"""
    block_model = omf.RegularBlockModel()
    block_model.definition.block_size = [1.0, 2.0, 3.0]
    with pytest.raises(properties.ValidationError):
        block_model.definition.block_count = block_count
        block_model.validate()


@pytest.mark.parametrize("block_size", ([2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [-1.0, 2, 2], [0.0, 2, 2]))
def test_bad_block_size(block_size):
    """Test mismatched block_size"""
    block_model = omf.RegularBlockModel()
    block_model.definition.block_count = [2, 2, 2]
    with pytest.raises(properties.ValidationError):
        block_model.definition.block_size = block_size
        block_model.validate()


def test_uninstantiated():
    """Test all attributes are None on instantiation"""
    block_model = omf.RegularBlockModel()
    assert block_model.definition.block_count is None
    assert block_model.definition.block_size is None
    assert block_model.num_cells is None


def test_num_cells():
    """Test num_cells calculation is correct"""
    block_model = omf.RegularBlockModel()
    block_model.definition.block_count = [2, 2, 2]
    block_model.definition.block_size = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(block_model.definition.block_count, [2, 2, 2])
    assert block_model.num_cells == 8
    assert block_model.location_length("cells") == 8
    assert block_model.location_length("parent_blocks") == 8
