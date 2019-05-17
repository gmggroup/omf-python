"""Tests for block models"""
import numpy as np
import properties
import pytest

import omf


def test_ijk_to_index():

    class BlockModelTester(omf.blockmodel.BaseBlockModel):

        num_parent_blocks = None

    block_model = BlockModelTester()
    with pytest.raises(AttributeError):
        block_model.ijk_to_index([0, 0, 0])
    block_model.num_parent_blocks = [3, 4, 5]
    with pytest.raises(ValueError):
        block_model.ijk_to_index('000')
    with pytest.raises(ValueError):
        block_model.ijk_to_index([0, 0])
    with pytest.raises(ValueError):
        block_model.ijk_to_index([0, 0, 0.5])
    with pytest.raises(ValueError):
        block_model.ijk_to_index([0, 0, 5])
    assert block_model.ijk_to_index([0, 0, 0]) == 0
    assert block_model.ijk_to_index([1, 0, 0]) == 1
    assert block_model.ijk_to_index([0, 1, 0]) == 3
    assert block_model.ijk_to_index([0, 0, 1]) == 12
    assert block_model.ijk_to_index([2, 3, 4]) == 59

    block_model = BlockModelTester()
    block_model.num_parent_blocks = [3, 4, 5]
    with pytest.raises(ValueError):
        block_model.ijk_array_to_indices('000')
    assert np.array_equal(block_model.ijk_array_to_indices([
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 3, 4)
    ]), [0, 1, 3, 12, 59])


def test_tensorblockmodel():
    """Test volume grid geometry validation"""
    elem = omf.TensorBlockModel()
    elem.tensor_u = [1., 1.]
    elem.tensor_v = [2., 2., 2.]
    elem.tensor_w = [3.]
    assert elem.validate()
    assert elem.location_length('vertices') == 24
    assert elem.location_length('cells') == 6
    elem.axis_v = [1., 1., 0]
    with pytest.raises(ValueError):
        elem.validate()
    elem.axis_v = 'Y'


class TestRegularBlockModel(object):

    bm_class = omf.RegularBlockModel

    @pytest.mark.parametrize(
        'num_blocks', ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5])
    )
    def test_bad_num_blocks(self, num_blocks):
        """Test mismatched num_blocks"""
        block_model = self.bm_class(size_blocks=[1., 2., 3.])
        with pytest.raises(properties.ValidationError):
            block_model.num_blocks = num_blocks
            block_model.validate()


    @pytest.mark.parametrize(
        'size_blocks', ([2., 2.], [2., 2., 2., 2.], [-1., 2, 2], [0., 2, 2])
    )
    def test_bad_size_blocks(self, size_blocks):
        """Test mismatched size_blocks"""
        block_model = self.bm_class(num_blocks=[2, 2, 2])
        with pytest.raises(properties.ValidationError):
            block_model.size_blocks = size_blocks
            block_model.validate()


    def test_uninstantiated(self):
        """Test all attributes are None on instantiation"""
        block_model = self.bm_class()
        assert block_model.num_blocks is None
        assert block_model.size_blocks is None
        assert block_model.cbc is None
        assert block_model.cbi is None


    def test_num_cells(self):
        """Test num_cells calculation is correct"""
        block_model = self.bm_class(
            num_blocks=[2, 2, 2],
            size_blocks=[1., 2., 3.],
        )
        assert block_model.num_cells == 8
        block_model.cbc = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        assert block_model.num_cells == 4


    def test_cbc(self):
        """Test cbc access and validation is correct"""
        block_model = self.bm_class(
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


    def test_cbi(self):
        """Test cbi access and validation is correct"""
        block_model = self.bm_class()
        assert block_model.cbi is None
        block_model.num_blocks = [2, 2, 2]
        block_model.size_blocks = [1., 2., 3.]
        assert np.all(block_model.cbi == np.array(range(9), dtype='int8'))
        block_model.cbc[0] = 0
        assert np.all(block_model.cbi == np.r_[np.array([0], dtype='int8'),
                                               np.array(range(8), dtype='int8')])


class TestRegularSubBlockModel(object):

    bm_class = omf.RegularSubBlockModel

    @pytest.mark.parametrize(
        'num_blocks', ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5])
    )
    @pytest.mark.parametrize('attr', ('num_parent_blocks', 'num_sub_blocks'))
    def test_bad_num_blocks(self, num_blocks, attr):
        """Test mismatched num_blocks"""
        block_model = self.bm_class(size_parent_blocks=[1., 2., 3.])
        with pytest.raises(properties.ValidationError):
            setattr(block_model, attr, num_blocks)
            block_model.validate()


    @pytest.mark.parametrize(
        'size_blocks', ([2., 2.], [2., 2., 2., 2.], [-1., 2, 2], [0., 2, 2])
    )
    def test_bad_size_blocks(self, size_blocks):
        """Test mismatched size_blocks"""
        block_model = self.bm_class(num_parent_blocks=[2, 2, 2])
        with pytest.raises(properties.ValidationError):
            block_model.size_parent_blocks = size_blocks
            block_model.validate()


    def test_uninstantiated(self):
        """Test all attributes are None on instantiation"""
        block_model = self.bm_class()
        assert block_model.num_parent_blocks is None
        assert block_model.num_sub_blocks is None
        assert block_model.size_parent_blocks is None
        assert block_model.size_sub_blocks is None
        assert block_model.cbc is None
        assert block_model.cbi is None


    def test_num_cells(self):
        """Test num_cells calculation is correct"""
        block_model = self.bm_class(
            num_parent_blocks=[2, 2, 2],
            num_sub_blocks=[2, 2, 2],
            size_parent_blocks=[1., 2., 3.],
        )
        assert block_model.num_cells == 8
        block_model.cbc = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        assert block_model.num_cells == 4
        block_model.refine([1, 1, 1])
        assert block_model.num_cells == 11


    def test_cbc(self):
        """Test cbc access and validation is correct"""
        block_model = self.bm_class(
            num_parent_blocks=[2, 2, 2],
            num_sub_blocks=[3, 4, 5],
            size_parent_blocks=[1., 2., 3.],
        )
        assert block_model.validate()
        assert np.all(block_model.cbc == np.ones(8))
        block_model.cbc[0] = 0
        assert block_model.validate()
        block_model.cbc[0] = 60
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

    def test_cbi(self):
        """Test cbi access and validation is correct"""
        block_model = self.bm_class()
        assert block_model.cbi is None
        block_model.num_parent_blocks = [2, 2, 2]
        block_model.num_sub_blocks = [3, 4, 5]
        block_model.size_parent_blocks = [1., 2., 3.]
        assert np.all(block_model.cbi == np.array(range(9), dtype='int8'))
        block_model.cbc[0] = 0
        assert np.all(
            block_model.cbi == np.r_[np.array([0], dtype='int8'),
                                     np.array(range(8), dtype='int8')]
        )
        block_model.refine([1, 0, 0])
        assert np.all(
            block_model.cbi == np.r_[np.array([0, 0], dtype='int8'),
                                     np.array(range(60, 67), dtype='int8')]
        )

    def test_location_length(self):
        block_model = self.bm_class(
            num_parent_blocks=[2, 2, 2],
            num_sub_blocks=[3, 4, 5],
            size_parent_blocks=[1., 2., 3.],
        )
        assert block_model.location_length('parent_blocks') == 8
        assert block_model.location_length('sub_blocks') == 8
        block_model.refine([0, 0, 0])
        assert block_model.location_length('parent_blocks') == 8
        assert block_model.location_length('sub_blocks') == 67

