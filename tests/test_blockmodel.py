"""Tests for block models"""
import numpy as np
import properties
import pytest

import omf


def test_ijk_to_index():
    """Test ijk indexing into parent blocks works as expected"""

    class BlockModelTester(omf.blockmodel.BaseBlockModel):
        """Dummy Block Model class for overriding num_parent_blocks"""

        num_parent_blocks = None

        def location_length(self, location):
            return 0

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
    """Test class for regular block model functionality"""

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
    """Test class for regular sub block model functionality"""

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
        """Ensure location length updates as expected with block refinement"""
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


class TestOctreeSubBlockModel(object):

    bm_class = omf.OctreeSubBlockModel

    @pytest.mark.parametrize(
        'num_blocks', ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5])
    )
    def test_bad_num_blocks(self, num_blocks):
        """Test mismatched num_blocks"""
        block_model = self.bm_class(size_parent_blocks=[1., 2., 3.])
        with pytest.raises(properties.ValidationError):
            block_model.size_parent_blocks = num_blocks
            block_model.validate()


    @pytest.mark.parametrize(
        'size_blocks', ([2., 2.], [2., 2., 2., 2.], [-1., 2, 2], [0., 2, 2])
    )
    def test_bad_size_blocks(self, size_blocks):
        """Test mismatched size_blocks"""
        block_model = self.bm_class(num_parent_blocks=[2, 2, 2])
        with pytest.raises(properties.ValidationError):
            block_model.num_parent_blocks = size_blocks
            block_model.validate()


    def test_uninstantiated(self):
        """Test all attributes are None on instantiation"""
        block_model = self.bm_class()
        assert block_model.num_parent_blocks is None
        assert block_model.size_parent_blocks is None
        assert block_model.cbc is None
        assert block_model.cbi is None
        assert block_model.zoc is None


    def test_num_cells(self):
        """Test num_cells calculation is correct"""
        block_model = self.bm_class(
            num_parent_blocks=[2, 2, 2],
            size_parent_blocks=[1., 2., 3.],
        )
        assert block_model.num_cells == 8
        block_model.cbc = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        assert block_model.num_cells == 4


    def test_cbc(self):
        """Test cbc access and validation is correct"""
        block_model = self.bm_class(
            num_parent_blocks=[2, 2, 2],
            size_parent_blocks=[1., 2., 3.],
        )
        assert block_model.validate()
        assert np.all(block_model.cbc == np.ones(8))
        block_model.cbc[0] = 0
        block_model.zoc = block_model.zoc[1:]
        assert block_model.validate()
        with pytest.raises(properties.ValidationError):
            block_model.cbc = np.ones(7, dtype='int8')
        block_model.cbc = np.ones(8, dtype='uint8')
        block_model.zoc = np.zeros(8, dtype='uint8')
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
        block_model.size_parent_blocks = [1., 2., 3.]
        assert np.all(block_model.cbi == np.array(range(9), dtype=np.uint64))
        block_model.cbc[0] = 0
        assert np.all(block_model.cbi == np.r_[np.array([0], dtype=np.uint64),
                                               np.array(range(8), dtype=np.uint64)])


    def test_zoc(self):
        block_model = self.bm_class(
            num_parent_blocks=[2, 2, 2],
            size_parent_blocks=[1., 2., 3.],
        )
        assert np.all(block_model.zoc == np.zeros(8))
        with pytest.raises(properties.ValidationError):
            block_model.zoc = np.zeros(7, dtype=np.uint64)
        with pytest.raises(properties.ValidationError):
            block_model.zoc = np.r_[np.zeros(7), -1.].astype(np.uint64)
        with pytest.raises(properties.ValidationError):
            block_model.zoc = np.r_[np.zeros(7), 268435448+1].astype(np.uint64)
        block_model.zoc = np.r_[np.zeros(7), 268435448].astype(np.uint64)
        assert block_model.validate()


    @pytest.mark.parametrize(
        ('pointer', 'level', 'curve_value'), [
            ([1, 16, 0], 7, 131095),
            ([0, 0, 0], 0, 0),
            ([255, 255, 255], 8, 268435448),
        ]
    )
    def test_curve_values(self, pointer, level, curve_value):
        assert self.bm_class.get_curve_value(pointer, level) == curve_value
        assert self.bm_class.get_level(curve_value) == level
        assert self.bm_class.get_pointer(curve_value) == pointer


    def test_refinement(self):
        block_model = self.bm_class(
            num_parent_blocks=[2, 2, 2],
            size_parent_blocks=[5., 5., 5.],
        )
        assert len(block_model.zoc) == 8
        assert all(zoc == 0 for zoc in block_model.zoc)
        block_model.refine(0)
        assert len(block_model.zoc) == 15
        assert np.array_equal(block_model.cbc, [8] + [1]*7)
        assert np.array_equal(block_model.cbi, [0] + list(range(8, 16)))
        assert np.array_equal(
            block_model.zoc, [
                block_model.get_curve_value([0, 0, 0], 1),
                block_model.get_curve_value([128, 0, 0], 1),
                block_model.get_curve_value([0, 128, 0], 1),
                block_model.get_curve_value([128, 128, 0], 1),
                block_model.get_curve_value([0, 0, 128], 1),
                block_model.get_curve_value([128, 0, 128], 1),
                block_model.get_curve_value([0, 128, 128], 1),
                block_model.get_curve_value([128, 128, 128], 1),
            ] + [0]*7
        )
        block_model.refine(2, refinements=2)
        assert len(block_model.zoc) == 78
        assert np.array_equal(block_model.cbc, [71] + [1]*7)
        assert np.array_equal(block_model.cbi, [0] + list(range(71, 79)))
        assert block_model.zoc[2] == block_model.get_curve_value([0, 128, 0], 3)
        assert block_model.zoc[3] == block_model.get_curve_value([32, 128, 0], 3)
        assert block_model.zoc[4] == block_model.get_curve_value([0, 160, 0], 3)
        assert block_model.zoc[5] == block_model.get_curve_value([32, 160, 0], 3)
        assert block_model.zoc[6] == block_model.get_curve_value([0, 128, 32], 3)
        assert block_model.zoc[64] == block_model.get_curve_value([64, 224, 96], 3)
        assert block_model.zoc[65] == block_model.get_curve_value([96, 224, 96], 3)
        assert block_model.zoc[66] == block_model.get_curve_value([128, 128, 0], 1)
        block_model.refine(0, [1, 0, 0])
        assert len(block_model.zoc) == 85
        assert np.array_equal(block_model.cbc, [71, 8] + [1]*6)
        with pytest.raises(ValueError):
            block_model.refine(85)
        with pytest.raises(ValueError):
            block_model.refine(-1)
        with pytest.raises(ValueError):
            block_model.refine(1, [1, 1, 1])
        with pytest.raises(ValueError):
            block_model.refine(2, refinements=-1)
        with pytest.raises(ValueError):
            block_model.refine(2, refinements=6)
