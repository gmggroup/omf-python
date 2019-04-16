"""Tests for block models"""
import numpy as np
import properties
import pytest

import omf


@pytest.mark.parametrize(
    'num_blocks', ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5])
)
def test_bad_num_blocks(num_blocks):
    """Test mismatched num_blocks"""
    block_model = omf.RegularBlockModel(size_blocks=[1., 2., 3.])
    with pytest.raises(properties.ValidationError):
        block_model.num_blocks = num_blocks
        block_model.validate()


@pytest.mark.parametrize(
    'size_blocks', ([2., 2.], [2., 2., 2., 2.], [-1., 2, 2])
)
def test_bad_size_blocks(size_blocks):
    """Test mismatched size_blocks"""
    block_model = omf.RegularBlockModel(num_blocks=[2, 2, 2])
    with pytest.raises(properties.ValidationError):
        block_model.size_blocks = size_blocks
        block_model.validate()


def test_uninstantiated():
    """Test all attributes are None on instantiation"""
    block_model = omf.RegularBlockModel()
    assert block_model.num_blocks is None
    assert block_model.size_blocks is None
    assert block_model.cbc is None
    assert block_model.cbc is None
    assert block_model.cbi is None


def test_num_cells():
    """Test num_cells calculation is correct"""
    block_model = omf.RegularBlockModel(
        num_blocks=[2, 2, 2],
        size_blocks=[1., 2., 3.],
    )
    assert block_model.num_cells == 8
    block_model.cbc = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert block_model.num_cells == 4


def test_cbc():
    """Test cbc access and validation is correct"""
    block_model = omf.RegularBlockModel(
        num_blocks=[2, 2, 2],
        size_blocks=[1., 2., 3.],
    )
    assert block_model.validate()
    assert np.all(block_model.cbc == np.ones(8))
    block_model.cbc[0] = 0
    assert block_model.validate()
    with pytest.raises(properties.ValidationError):
        block_model.cbc = np.ones(7, dtype='int8')
    block_model.cbc = np.ones(8, dtype='uint8')
    with pytest.raises(properties.ValidationError):
        block_model.cbc[0] = 2
        block_model.validate()
    with pytest.raises(properties.ValidationError):
        block_model.cbc[0] = -1
        block_model.validate()


def test_cbi():
    """Test cbi access and validation is correct"""
    block_model = omf.RegularBlockModel()
    assert block_model.cbi is None
    block_model.num_blocks = [2, 2, 2]
    block_model.size_blocks = [1., 2., 3.]
    assert np.all(block_model.cbi == np.array(range(9), dtype='int8'))
    block_model.cbc[0] = 0
    assert np.all(block_model.cbi == np.r_[np.array([0], dtype='int8'),
                                           np.array(range(8), dtype='int8')])
