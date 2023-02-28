"""Tests for block models"""
import numpy as np
import properties
import pytest

import omf


def _make_regular(count):
    bm = omf.RegularBlockModel()
    bm.definition.block_count = count
    bm.definition.block_size = [1.0, 1.0, 1.0]
    return bm


class MockArray(omf.base.BaseModel):
    """Test array class"""

    array = np.array([1, 2, 3])


def test_ijk_index_errors():
    """Test ijk indexing into parent blocks errors as expected"""

    block_model = _make_regular([3, 4, 5])
    with pytest.raises(TypeError):
        block_model.definition.ijk_to_index("a")
    with pytest.raises(TypeError):
        block_model.definition.index_to_ijk("a")
    with pytest.raises(ValueError):
        block_model.definition.ijk_to_index([0, 0])
    with pytest.raises(TypeError):
        block_model.definition.ijk_to_index([0, 0, 0.5])
    with pytest.raises(TypeError):
        block_model.definition.index_to_ijk(0.5)
    with pytest.raises(IndexError):
        block_model.definition.ijk_to_index([0, 0, 5])
    with pytest.raises(IndexError):
        block_model.definition.index_to_ijk(60)
    with pytest.raises(IndexError):
        block_model.definition.ijk_to_index([[0, 0, 5], [0, 0, 3]])
    with pytest.raises(IndexError):
        block_model.definition.index_to_ijk([0, 1, 60])


def test_ijk_index_arrays():
    """Test ijk array indexing into parent blocks works as expected"""
    block_model = _make_regular([3, 4, 5])
    ijk = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 3, 4)]
    index = [0, 1, 3, 12, 59]
    assert np.array_equal(block_model.definition.ijk_to_index(ijk), index)
    assert np.array_equal(block_model.definition.index_to_ijk(index), ijk)
    ijk = [[(0, 0, 0), (1, 0, 0)], [(0, 1, 0), (0, 0, 1)]]
    index = [(0, 1), (3, 12)]
    assert np.array_equal(block_model.definition.ijk_to_index(ijk), index)
    assert np.array_equal(block_model.definition.index_to_ijk(index), ijk)


@pytest.mark.parametrize(
    ("ijk", "index"),
    [([0, 0, 0], 0), ([1, 0, 0], 1), ([0, 1, 0], 3), ([0, 0, 1], 12), ([2, 3, 4], 59)],
)
def test_ijk_index(ijk, index):
    """Test ijk indexing into parent blocks works as expected"""
    block_model = _make_regular([3, 4, 5])
    assert block_model.definition.ijk_to_index(ijk) == index
    assert np.array_equal(block_model.definition.index_to_ijk(index), ijk)


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


# pylint: disable=W0143
class TestRegularBlockModel:
    """Test class for regular block model functionality"""

    @pytest.mark.parametrize(
        "block_count", ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5])
    )
    def test_bad_block_count(self, block_count):
        """Test mismatched block_count"""
        block_model = omf.RegularBlockModel()
        block_model.definition.block_size = [1.0, 2.0, 3.0]
        with pytest.raises(properties.ValidationError):
            block_model.definition.block_count = block_count
            block_model.validate()

    @pytest.mark.parametrize(
        "block_size", ([2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [-1.0, 2, 2], [0.0, 2, 2])
    )
    def test_bad_block_size(self, block_size):
        """Test mismatched block_size"""
        block_model = omf.RegularBlockModel()
        block_model.definition.block_count = [2, 2, 2]
        with pytest.raises(properties.ValidationError):
            block_model.definition.block_size = block_size
            block_model.validate()

    def test_uninstantiated(self):
        """Test all attributes are None on instantiation"""
        block_model = omf.RegularBlockModel()
        assert block_model.definition.block_count is None
        assert block_model.definition.block_size is None
        assert block_model.num_cells is None

    def test_num_cells(self):
        """Test num_cells calculation is correct"""
        block_model = omf.RegularBlockModel()
        block_model.definition.block_count = [2, 2, 2]
        block_model.definition.block_size = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(block_model.definition.block_count, [2, 2, 2])
        assert block_model.num_cells == 8
        assert block_model.location_length("cells") == 8
        assert block_model.location_length("parent_blocks") == 8


class TestSubblockedModel:
    def test_pack_uints(self):
        block_model = omf.SubblockedModel()
        block_model.subblock_definition.count = [2, 2, 2]
        block_model.definition.block_size = [1.0, 1.0, 1.0]
        block_model.definition.block_count = [10, 10, 10]
        block_model.subblock_parent_indices = np.array([(0, 0, 0)])
        block_model.subblock_corners = np.array([(0, 0, 0, 2, 2, 2)])
        # We set this as uint32
        assert block_model.subblock_corners.dtype == np.int32
        block_model.validate()
        # Validate should have packed it down to uint8
        assert block_model.subblock_corners.dtype == np.uint8


# class TestRegularSubBlockModel:
#     """Test class for regular sub block model functionality"""

#     bm_class = omf.RegularSubBlockModel

#     @pytest.mark.parametrize(
#         "block_count", ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5])
#     )
#     @pytest.mark.parametrize("attr", ("parent_block_count", "sub_block_count"))
#     def test_bad_block_count(self, block_count, attr):
#         """Test mismatched block_count"""
#         block_model = self.bm_class(parent_block_size=[1.0, 2.0, 3.0])
#         with pytest.raises(properties.ValidationError):
#             setattr(block_model, attr, block_count)
#             block_model.validate()

#     @pytest.mark.parametrize(
#         "block_size", ([2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [-1.0, 2, 2], [0.0, 2, 2])
#     )
#     def test_bad_block_size(self, block_size):
#         """Test mismatched block_size"""
#         block_model = self.bm_class(parent_block_count=[2, 2, 2])
#         with pytest.raises(properties.ValidationError):
#             block_model.parent_block_size = block_size
#             block_model.validate()

#     def test_uninstantiated(self):
#         """Test all attributes are None on instantiation"""
#         block_model = self.bm_class()
#         assert block_model.parent_block_count is None
#         assert block_model.sub_block_count is None
#         assert block_model.parent_block_size is None
#         assert block_model.sub_block_size is None
#         assert block_model.cbc is None
#         assert block_model.cbi is None
#         assert block_model.num_cells is None
#         with pytest.raises(ValueError):
#             block_model.reset_cbc()
#         with pytest.raises(ValueError):
#             block_model.refine([0, 0, 0])
#         block_model.validate_cbc({"value": MockArray()})

#     def test_num_cells(self):
#         """Test num_cells calculation is correct"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             sub_block_count=[2, 2, 2],
#             parent_block_size=[1.0, 2.0, 3.0],
#         )
#         block_model.reset_cbc()
#         assert block_model.num_cells == 8
#         block_model.cbc = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#         assert block_model.num_cells == 4
#         block_model.refine([1, 1, 1])
#         assert block_model.num_cells == 11

#     def test_cbc(self):
#         """Test cbc access and validation is correct"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             sub_block_count=[3, 4, 5],
#             parent_block_size=[1.0, 2.0, 3.0],
#         )
#         block_model.reset_cbc()
#         assert block_model.validate()
#         assert np.all(block_model.cbc == np.ones(8))
#         block_model.cbc.array[0] = 0
#         assert block_model.validate()
#         block_model.cbc.array[0] = 60
#         assert block_model.validate()
#         with pytest.raises(properties.ValidationError):
#             block_model.cbc = np.ones(7, dtype="int8")
#         block_model.cbc = np.ones(8, dtype="uint8")
#         with pytest.raises(properties.ValidationError):
#             block_model.cbc.array[0] = 2
#             block_model.validate()
#         with pytest.raises(properties.ValidationError):
#             block_model.cbc.array[0] = -1
#             block_model.validate()

#     def test_cbi(self):
#         """Test cbi access and validation is correct"""
#         block_model = self.bm_class()
#         assert block_model.cbi is None
#         block_model.parent_block_count = [2, 2, 2]
#         block_model.sub_block_count = [3, 4, 5]
#         block_model.parent_block_size = [1.0, 2.0, 3.0]
#         block_model.reset_cbc()
#         assert np.all(block_model.cbi == np.array(range(9), dtype="int8"))
#         block_model.cbc.array[0] = 0
#         assert np.all(
#             block_model.cbi
#             == np.r_[np.array([0], dtype="int8"), np.array(range(8), dtype="int8")]
#         )
#         block_model.refine([1, 0, 0])
#         assert np.all(
#             block_model.cbi
#             == np.r_[
#                 np.array([0, 0], dtype="int8"), np.array(range(60, 67), dtype="int8")
#             ]
#         )

#     def test_location_length(self):
#         """Ensure location length updates as expected with block refinement"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             sub_block_count=[3, 4, 5],
#             parent_block_size=[1.0, 2.0, 3.0],
#         )
#         block_model.reset_cbc()
#         assert block_model.location_length("parent_blocks") == 8
#         assert block_model.location_length("sub_blocks") == 8
#         block_model.refine([0, 0, 0])
#         assert block_model.location_length("parent_blocks") == 8
#         assert block_model.location_length("sub_blocks") == 67


# class TestOctreeSubBlockModel:
#     """Test class for octree sub block model"""

#     bm_class = omf.OctreeSubBlockModel

#     @pytest.mark.parametrize(
#         "block_count", ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5])
#     )
#     def test_bad_block_count(self, block_count):
#         """Test mismatched block_count"""
#         block_model = self.bm_class(parent_block_size=[1.0, 2.0, 3.0])
#         with pytest.raises(properties.ValidationError):
#             block_model.parent_block_size = block_count
#             block_model.validate()

#     @pytest.mark.parametrize(
#         "block_size", ([2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [-1.0, 2, 2], [0.0, 2, 2])
#     )
#     def test_bad_block_size(self, block_size):
#         """Test mismatched block_size"""
#         block_model = self.bm_class(parent_block_count=[2, 2, 2])
#         with pytest.raises(properties.ValidationError):
#             block_model.parent_block_count = block_size
#             block_model.validate()

#     def test_uninstantiated(self):
#         """Test all attributes are None on instantiation"""
#         block_model = self.bm_class()
#         assert block_model.parent_block_count is None
#         assert block_model.parent_block_size is None
#         assert block_model.cbc is None
#         assert block_model.cbi is None
#         assert block_model.zoc is None
#         assert block_model.num_cells is None
#         block_model.validate_cbc({"value": MockArray()})
#         block_model.validate_zoc({"value": MockArray()})
#         with pytest.raises(ValueError):
#             block_model.reset_cbc()
#         with pytest.raises(ValueError):
#             block_model.reset_zoc()

#     def test_num_cells(self):
#         """Test num_cells calculation is correct"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             parent_block_size=[1.0, 2.0, 3.0],
#         )
#         block_model.reset_cbc()
#         assert block_model.num_cells == 8
#         block_model.cbc = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#         assert block_model.num_cells == 4

#     def test_cbc(self):
#         """Test cbc access and validation is correct"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             parent_block_size=[1.0, 2.0, 3.0],
#         )
#         block_model.reset_cbc()
#         block_model.reset_zoc()
#         assert block_model.validate()
#         assert np.all(block_model.cbc == np.ones(8))
#         block_model.cbc.array[0] = 0
#         block_model.zoc = block_model.zoc[1:]
#         assert block_model.validate()
#         with pytest.raises(properties.ValidationError):
#             block_model.cbc = np.ones(7, dtype="int8")
#         block_model.cbc = np.ones(8, dtype="uint8")
#         block_model.zoc = np.zeros(8, dtype="uint8")
#         assert block_model.validate()
#         with pytest.raises(properties.ValidationError):
#             block_model.cbc.array[0] = 2
#             block_model.validate()
#         with pytest.raises(properties.ValidationError):
#             block_model.cbc.array[0] = -1
#             block_model.validate()

#     def test_cbi(self):
#         """Test cbi access and validation is correct"""
#         block_model = self.bm_class()
#         assert block_model.cbi is None
#         block_model.parent_block_count = [2, 2, 2]
#         block_model.parent_block_size = [1.0, 2.0, 3.0]
#         block_model.reset_cbc()
#         assert np.all(block_model.cbi == np.array(range(9), dtype=np.uint64))
#         block_model.cbc.array[0] = 0
#         assert np.all(
#             block_model.cbi
#             == np.r_[
#                 np.array([0], dtype=np.uint64), np.array(range(8), dtype=np.uint64)
#             ]
#         )

#     def test_zoc(self):
#         """Test z-order curves"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             parent_block_size=[1.0, 2.0, 3.0],
#         )
#         block_model.reset_cbc()
#         block_model.reset_zoc()
#         assert np.all(block_model.zoc == np.zeros(8))
#         with pytest.raises(properties.ValidationError):
#             block_model.zoc = np.zeros(7, dtype=np.uint64)
#         with pytest.raises(properties.ValidationError):
#             block_model.zoc = np.r_[np.zeros(7), -1.0].astype(np.uint64)
#         with pytest.raises(properties.ValidationError):
#             block_model.zoc = np.r_[np.zeros(7), 268435448 + 1].astype(np.uint64)
#         block_model.zoc = np.r_[np.zeros(7), 268435448].astype(np.uint64)
#         assert block_model.validate()

#     @pytest.mark.parametrize(
#         ("pointer", "level", "curve_value"),
#         [
#             ([1, 16, 0], 7, 131095),
#             ([0, 0, 0], 0, 0),
#             ([255, 255, 255], 8, 268435448),
#         ],
#     )
#     def test_curve_values(self, pointer, level, curve_value):
#         """Test curve value functions"""
#         assert self.bm_class.get_curve_value(pointer, level) == curve_value
#         assert self.bm_class.get_level(curve_value) == level
#         assert self.bm_class.get_pointer(curve_value) == pointer

#     def test_level_width(self):
#         """Test level width function"""
#         with pytest.raises(ValueError):
#             self.bm_class.level_width(9)

#     def test_refinement(self):
#         """Test refinement method"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             parent_block_size=[5.0, 5.0, 5.0],
#         )
#         block_model.reset_cbc()
#         block_model.reset_zoc()
#         assert len(block_model.zoc) == 8
#         assert all(zoc == 0 for zoc in block_model.zoc)
#         block_model.refine(0)
#         assert len(block_model.zoc) == 15
#         assert block_model.location_length("parent_blocks") == 8
#         assert block_model.location_length("") == 15
#         assert np.array_equal(block_model.cbc, [8] + [1] * 7)
#         assert np.array_equal(block_model.cbi, [0] + list(range(8, 16)))
#         assert np.array_equal(
#             block_model.zoc,
#             [
#                 block_model.get_curve_value([0, 0, 0], 1),
#                 block_model.get_curve_value([128, 0, 0], 1),
#                 block_model.get_curve_value([0, 128, 0], 1),
#                 block_model.get_curve_value([128, 128, 0], 1),
#                 block_model.get_curve_value([0, 0, 128], 1),
#                 block_model.get_curve_value([128, 0, 128], 1),
#                 block_model.get_curve_value([0, 128, 128], 1),
#                 block_model.get_curve_value([128, 128, 128], 1),
#             ]
#             + [0] * 7,
#         )
#         block_model.refine(2, refinements=2)
#         assert len(block_model.zoc) == 78
#         assert np.array_equal(block_model.cbc, [71] + [1] * 7)
#         assert np.array_equal(block_model.cbi, [0] + list(range(71, 79)))
#         assert block_model.zoc[2] == block_model.get_curve_value([0, 128, 0], 3)
#         assert block_model.zoc[3] == block_model.get_curve_value([32, 128, 0], 3)
#         assert block_model.zoc[4] == block_model.get_curve_value([0, 160, 0], 3)
#         assert block_model.zoc[5] == block_model.get_curve_value([32, 160, 0], 3)
#         assert block_model.zoc[6] == block_model.get_curve_value([0, 128, 32], 3)
#         assert block_model.zoc[64] == block_model.get_curve_value([64, 224, 96], 3)
#         assert block_model.zoc[65] == block_model.get_curve_value([96, 224, 96], 3)
#         assert block_model.zoc[66] == block_model.get_curve_value([128, 128, 0], 1)
#         block_model.refine(0, [1, 0, 0])
#         assert len(block_model.zoc) == 85
#         assert np.array_equal(block_model.cbc, [71, 8] + [1] * 6)
#         with pytest.raises(ValueError):
#             block_model.refine(85)
#         with pytest.raises(ValueError):
#             block_model.refine(-1)
#         with pytest.raises(ValueError):
#             block_model.refine(1, [1, 1, 1])
#         with pytest.raises(ValueError):
#             block_model.refine(2, refinements=-1)
#         with pytest.raises(ValueError):
#             block_model.refine(2, refinements=6)


# class TestArbitrarySubBlockModel:
#     """Test class for ArbitrarySubBlockModel"""

#     bm_class = omf.ArbitrarySubBlockModel

#     @pytest.mark.parametrize(
#         "block_count", ([2, 2], [2, 2, 2, 2], [0, 2, 2], [2, 2, 0.5])
#     )
#     def test_bad_block_count(self, block_count):
#         """Test mismatched block_count"""
#         block_model = self.bm_class(parent_block_size=[1.0, 2.0, 3.0])
#         with pytest.raises(properties.ValidationError):
#             block_model.parent_block_size = block_count
#             block_model.validate()

#     @pytest.mark.parametrize(
#         "block_size", ([2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [-1.0, 2, 2], [0.0, 2, 2])
#     )
#     def test_bad_block_size(self, block_size):
#         """Test mismatched block_size"""
#         block_model = self.bm_class(parent_block_count=[2, 2, 2])
#         with pytest.raises(properties.ValidationError):
#             block_model.parent_block_count = block_size
#             block_model.validate()

#     def test_uninstantiated(self):
#         """Test all attributes are None on instantiation"""
#         block_model = self.bm_class()
#         assert block_model.parent_block_count is None
#         assert block_model.parent_block_size is None
#         assert block_model.cbc is None
#         assert block_model.cbi is None
#         assert block_model.sub_block_corners is None
#         assert block_model.sub_block_sizes is None
#         assert block_model.sub_block_centroids is None
#         assert block_model.sub_block_corners_absolute is None
#         assert block_model.sub_block_sizes_absolute is None
#         assert block_model.sub_block_centroids_absolute is None
#         assert block_model.num_cells is None
#         block_model.validate_cbc({"value": MockArray()})
#         with pytest.raises(ValueError):
#             block_model.reset_cbc()

#     def test_num_cells(self):
#         """Test num_cells calculation is correct"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             parent_block_size=[1.0, 2.0, 3.0],
#         )
#         block_model.reset_cbc()
#         assert block_model.num_cells == 8
#         block_model.cbc = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#         assert block_model.num_cells == 4

#     def test_cbc(self):
#         """Test cbc access and validation is correct"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             parent_block_size=[1.0, 2.0, 3.0],
#         )
#         with pytest.raises(properties.ValidationError):
#             block_model.validate()
#         block_model.sub_block_corners = np.zeros((8, 3))
#         block_model.sub_block_sizes = np.ones((8, 3))
#         block_model.reset_cbc()
#         assert block_model.validate()
#         assert np.all(block_model.cbc == np.ones(8))
#         block_model.cbc.array[0] = 0
#         with pytest.raises(properties.ValidationError):
#             block_model.validate()
#         block_model.sub_block_corners = np.zeros((7, 3))
#         block_model.sub_block_sizes = np.ones((7, 3))
#         assert block_model.validate()
#         with pytest.raises(properties.ValidationError):
#             block_model.cbc = np.ones(7, dtype="int8")
#         block_model.cbc = np.ones(8, dtype="uint8")
#         block_model.sub_block_corners = np.zeros((8, 3))
#         block_model.sub_block_sizes = np.ones((8, 3))
#         with pytest.raises(properties.ValidationError):
#             block_model.cbc.array[0] = 2
#             block_model.validate()
#         with pytest.raises(properties.ValidationError):
#             block_model.cbc.array[0] = -1
#             block_model.validate()

#     def test_cbi(self):
#         """Test cbi access and validation is correct"""
#         block_model = self.bm_class()
#         assert block_model.cbi is None
#         block_model.parent_block_count = [2, 2, 2]
#         block_model.parent_block_size = [1.0, 2.0, 3.0]
#         block_model.reset_cbc()
#         assert np.all(block_model.cbi == np.array(range(9), dtype=np.uint64))
#         block_model.cbc.array[0] = 0
#         assert np.all(
#             block_model.cbi
#             == np.r_[
#                 np.array([0], dtype=np.uint64), np.array(range(8), dtype=np.uint64)
#             ]
#         )

#     def test_validate_sub_block_attrs(self):
#         """Test sub block attribute validation"""
#         block_model = self.bm_class()
#         value = [1, 2, 3]
#         assert block_model.validate_sub_block_attributes(value, "") is value
#         block_model.parent_block_count = [2, 2, 2]
#         block_model.parent_block_size = [1.0, 2.0, 3.0]
#         block_model.reset_cbc()
#         with pytest.raises(properties.ValidationError):
#             block_model.validate_sub_block_attributes(value, "")

#     def test_validate_sub_block_sizes(self):
#         """Test sub block size validation"""
#         block_model = self.bm_class()
#         block_model.sub_block_sizes = [[1.0, 2, 3]]
#         with pytest.raises(properties.ValidationError):
#             block_model.sub_block_sizes = [[0.0, 1, 2]]

#     def test_sub_block_attributes(self):
#         """Test sub block attributes"""
#         block_model = self.bm_class(
#             parent_block_count=[2, 2, 2],
#             parent_block_size=[1.0, 2.0, 3.0],
#         )
#         block_model.reset_cbc()
#         with pytest.raises(properties.ValidationError):
#             block_model.sub_block_sizes = np.ones((3, 3))
#         with pytest.raises(properties.ValidationError):
#             block_model.sub_block_sizes = np.r_[np.ones((7, 3)), [[1.0, 1.0, 0]]]
#         block_model.sub_block_sizes = np.ones((8, 3))
#         assert np.array_equal(
#             block_model.sub_block_sizes_absolute, np.array([[1.0, 2.0, 3.0]] * 8)
#         )
#         assert block_model.sub_block_centroids is None
#         assert block_model.sub_block_centroids_absolute is None
#         with pytest.raises(properties.ValidationError):
#             block_model.sub_block_corners = np.zeros((3, 3))
#         block_model.sub_block_corners = np.zeros((8, 3))
#         assert np.array_equal(
#             block_model.sub_block_corners_absolute,
#             np.array(
#                 [
#                     [0.0, 0, 0],
#                     [1.0, 0, 0],
#                     [0.0, 2, 0],
#                     [1.0, 2, 0],
#                     [0.0, 0, 3],
#                     [1.0, 0, 3],
#                     [0.0, 2, 3],
#                     [1.0, 2, 3],
#                 ]
#             ),
#         )
#         assert np.array_equal(block_model.sub_block_centroids, np.ones((8, 3)) * 0.5)
#         assert np.array_equal(
#             block_model.sub_block_centroids_absolute,
#             np.array(
#                 [
#                     [0.5, 1, 1.5],
#                     [1.5, 1, 1.5],
#                     [0.5, 3, 1.5],
#                     [1.5, 3, 1.5],
#                     [0.5, 1, 4.5],
#                     [1.5, 1, 4.5],
#                     [0.5, 3, 4.5],
#                     [1.5, 3, 4.5],
#                 ]
#             ),
#         )
#         assert block_model.validate()
#         assert block_model.location_length("parent_blocks") == 8
#         assert block_model.location_length("") == 8
#         block_model.cbc = np.array([1] + [0] * 7, dtype=int)
#         with pytest.raises(properties.ValidationError):
#             block_model.validate()
#         block_model.sub_block_corners = np.array([[-0.5, 2, 0]])
#         block_model.sub_block_sizes = np.array([[0.5, 0.5, 2]])
#         assert block_model.validate()
#         assert block_model.location_length("parent_blocks") == 1
#         assert block_model.location_length("") == 1
#         assert np.array_equal(
#             block_model.sub_block_centroids, np.array([[-0.25, 2.25, 1]])
#         )
#         assert np.array_equal(
#             block_model.sub_block_corners_absolute, np.array([[-0.5, 4, 0]])
#         )
#         assert np.array_equal(
#             block_model.sub_block_sizes_absolute, np.array([[0.5, 1, 6]])
#         )
#         assert np.array_equal(
#             block_model.sub_block_centroids_absolute, np.array([[-0.25, 4.5, 3]])
#         )
#         assert block_model.validate()


# pylint: enable=W0143
