"""Tests for block models"""
import numpy as np
import properties
import pytest

import omf

@pytest.mark.parametrize('num_blocks', ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5]))
def test_bad_num_blocks(num_blocks):
    bm = omf.RegularBlockModel(size_blocks=[1., 2., 3.])
    with pytest.raises(properties.ValidationError):
        bm.num_blocks = num_blocks
        bm.validate()


@pytest.mark.parametrize('size_blocks', ([2., 2.], [2., 2., 2., 2.], [-1., 2, 2]))
def test_bad_num_blocks(size_blocks):
    bm = omf.RegularBlockModel(num_blocks=[2, 2, 2])
    with pytest.raises(properties.ValidationError):
        bm.size_blocks = size_blocks
        bm.validate()


def test_uninstantiated():
    bm = omf.RegularBlockModel()
    assert bm.num_blocks is None
    assert bm.size_blocks is None
    assert bm.cbc is None
    assert bm.cbc is None
    assert bm.cbi is None


def test_num_cells():
    bm = omf.RegularBlockModel(
        num_blocks=[2, 2, 2],
        size_blocks=[1., 2., 3.],
    )
    assert bm.num_cells == 8
    bm.cbc = [0, 0, 0, 0, 1, 1, 1, 1]
    assert bm.num_cells == 4


def test_cbc():
    bm = omf.RegularBlockModel(
        num_blocks=[2, 2, 2],
        size_blocks=[1., 2., 3.],
    )
    assert bm.validate()
    assert np.all(bm.cbc == np.ones(8))
    bm.cbc[0] = 0
    assert bm.validate()
    with pytest.raises(properties.ValidationError):
        bm.cbc = np.ones(7, dtype='int8')
    with pytest.raises(properties.ValidationError):
        bm.cbc[0] = 2
        bm.validate()
    with pytest.raises(properties.ValidationError):
        bm.cbc[0] = -1
        bm.validate()


def test_cbi():
    bm = omf.RegularBlockModel()
    assert bm.cbi is None
    bm.num_blocks = [2, 2, 2]
    bm.size_blocks = [1., 2., 3.]
    assert np.all(bm.cbi == np.array(range(9), dtype='int8'))
    bm.cbc[0] = 0
    assert np.all(bm.cbi == np.r_[
        np.array([0], dtype='int8'), np.array(range(8), dtype='int8')
    ])
