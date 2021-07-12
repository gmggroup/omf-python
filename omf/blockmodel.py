"""blockmodel.py: Block Model element definitions"""
import numpy as np
import properties

from .base import ProjectElement
from .attribute import ArrayInstanceProperty


class BaseBlockModel(ProjectElement):
    """Basic orientation properties for all block models"""

    axis_u = properties.Vector3(
        "Vector orientation of u-direction",
        default="X",
        length=1,
    )
    axis_v = properties.Vector3(
        "Vector orientation of v-direction",
        default="Y",
        length=1,
    )
    axis_w = properties.Vector3(
        "Vector orientation of w-direction",
        default="Z",
        length=1,
    )
    corner = properties.Vector3(
        "Corner of the block model relative to Project coordinate reference system",
        default=[0.0, 0.0, 0.0],
    )

    @properties.validator
    def _validate_axes(self):
        """Check if mesh content is built correctly"""
        if not (
            np.abs(self.axis_u.dot(self.axis_v) < 1e-6)
            and np.abs(self.axis_v.dot(self.axis_w) < 1e-6)
            and np.abs(self.axis_w.dot(self.axis_u) < 1e-6)
        ):
            raise ValueError("axis_u, axis_v, and axis_w must be orthogonal")
        return True

    @property
    def parent_block_count(self):
        """Computed property for number of parent blocks

        This is required for ijk indexing. For tensor and regular block
        models, all the blocks are considered parent blocks.
        """
        raise NotImplementedError()

    def ijk_to_index(self, ijk):
        """Return index for single ijk triple"""
        return self.ijk_array_to_indices([ijk])[0]

    def ijk_array_to_indices(self, ijk_array):
        """Return an array of indices for a list of ijk triples"""
        blocks = self.parent_block_count
        if not blocks:
            raise AttributeError("parent_block_count is required to calculate index")
        if not isinstance(ijk_array, (list, tuple, np.ndarray)):
            raise ValueError("ijk_array must be a list of length-3 ijk values")
        ijk_array = np.array(ijk_array)
        if len(ijk_array.shape) != 2 or ijk_array.shape[1] != 3:
            raise ValueError("ijk_array must be n x 3 array")
        if not np.array_equal(ijk_array, ijk_array.astype(np.uint32)):
            raise ValueError("ijk values must be non-negative integers")
        if np.any(np.max(ijk_array, axis=0) >= blocks):
            raise ValueError(
                "ijk must be less than parent_block_count in each dimension"
            )
        index = np.ravel_multi_index(
            multi_index=ijk_array.T,
            dims=blocks,
            order="F",
        )
        return index

    def index_to_ijk(self, index):
        """Return ijk triple for single index"""
        return self.indices_to_ijk_array([index])[0]

    def indices_to_ijk_array(self, indices):
        """Return an array of ijk triples for an array of indices"""
        blocks = self.parent_block_count
        if not blocks:
            raise AttributeError(
                "parent_block_count is required to calculate ijk values"
            )
        if not isinstance(indices, (list, tuple, np.ndarray)):
            raise ValueError("indices must be a list of index values")
        indices = np.array(indices)
        if len(indices.shape) != 1:
            raise ValueError("indices must be 1D array")
        if not np.array_equal(indices, indices.astype(np.uint64)):
            raise ValueError("indices values must be non-negative integers")
        if np.max(indices) >= np.prod(blocks):
            raise ValueError("indices must be less than total number of parent blocks")
        ijk = np.unravel_index(
            indices=indices,
            shape=blocks,
            order="F",
        )
        ijk_array = np.c_[ijk[0], ijk[1], ijk[2]]
        return ijk_array


class TensorGridBlockModel(BaseBlockModel):
    """Block model with variable spacing in each dimension"""

    schema = "org.omf.v2.element.blockmodel.tensorgrid"

    tensor_u = properties.Array(
        "Tensor cell widths, u-direction",
        shape=("*",),
        dtype=float,
    )
    tensor_v = properties.Array(
        "Tensor cell widths, v-direction",
        shape=("*",),
        dtype=float,
    )
    tensor_w = properties.Array(
        "Tensor cell widths, w-direction",
        shape=("*",),
        dtype=float,
    )

    _valid_locations = ("vertices", "cells", "parent_blocks")

    def location_length(self, location):
        """Return correct attribute length based on location"""
        if location == "vertices":
            return self.num_nodes
        return self.num_cells

    def _tensors_defined(self):
        """Check if all tensors are defined"""
        tensors = [self.tensor_u, self.tensor_v, self.tensor_w]
        return all((tensor is not None for tensor in tensors))

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        if not self._tensors_defined():
            return None
        nodes = (
            (len(self.tensor_u) + 1)
            * (len(self.tensor_v) + 1)
            * (len(self.tensor_w) + 1)
        )
        return nodes

    @property
    def num_cells(self):
        """Number of cells"""
        if not self._tensors_defined():
            return None
        cells = len(self.tensor_u) * len(self.tensor_v) * len(self.tensor_w)
        return cells

    @property
    def parent_block_count(self):
        """Number of parent blocks equals number of blocks"""
        if not self._tensors_defined():
            return None
        blocks = [len(self.tensor_u), len(self.tensor_v), len(self.tensor_w)]
        return blocks


class RegularBlockModel(BaseBlockModel):
    """Block model with constant spacing in each dimension"""

    schema = "org.omf.v2.elements.blockmodel.regular"

    block_count = properties.List(
        "Number of blocks along u, v, and w axes",
        properties.Integer("", min=1),
        min_length=3,
        max_length=3,
    )
    block_size = properties.List(
        "Size of blocks in the u, v, and w dimensions",
        properties.Float("", min=0),
        min_length=3,
        max_length=3,
    )
    cbc = ArrayInstanceProperty(
        "Compressed block count - for regular block models this must "
        "have length equal to the product of block_count and all values "
        "must be 1 (if attributes exist on the block) or 0; the default "
        "is an array of 1s",
        shape=("*",),
        dtype=(int, bool),
    )

    _valid_locations = ("cells", "parent_blocks")

    @properties.Array(
        "Compressed block index - used for indexing attributes "
        "into the block model; must have length equal to the "
        "product of block_count plus 1 and monotonically increasing",
        shape=("*",),
        dtype=int,
        coerce=False,
    )
    def cbi(self):
        """Compressed block index"""
        if self.cbc is None:
            return None
        # Recalculating the sum on the fly is faster than checking md5
        cbi = np.concatenate(
            [
                np.array([0], dtype=np.uint32),
                np.cumsum(self.cbc, dtype=np.uint32),
            ]
        )
        return cbi

    @properties.validator("block_size")
    def _validate_size_is_not_zero(self, change):
        """Ensure block sizes are non-zero"""
        if 0 in change["value"]:
            raise properties.ValidationError(
                "Block size cannot be 0",
                prop="block_size",
                instance=self,
                reason="invalid",
            )

    @properties.validator("cbc")
    def validate_cbc(self, change):
        """Ensure cbc is correct size and values"""
        value = change["value"]
        if self.block_count and len(value.array) != np.prod(self.block_count):
            raise properties.ValidationError(
                "cbc must have length equal to the product of block_count",
                prop="cbc",
                instance=self,
                reason="invalid",
            )
        if np.max(value.array) > 1 or np.min(value.array) < 0:
            raise properties.ValidationError(
                "cbc must have only values 0 or 1",
                prop="cbc",
                instance=self,
                reason="invalid",
            )

    @property
    def num_cells(self):
        """Number of cells from last value in the compressed block index"""
        cbi = self.cbi
        if cbi is None:
            return None
        return cbi[-1]  # pylint: disable=E1136

    def location_length(self, location):
        """Return correct attribute length based on location"""
        return self.num_cells

    @property
    def parent_block_count(self):
        """Number of parent blocks equals number of blocks"""
        return self.block_count

    def reset_cbc(self):
        """Reset cbc to no sub-blocks"""
        if not self.block_count:
            raise ValueError("cannot reset cbc until block_count is set")
        cbc_len = np.prod(self.block_count)
        self.cbc = np.ones(cbc_len, dtype=np.bool)


class RegularSubBlockModel(BaseBlockModel):
    """Block model with one level of sub-blocking possible in each parent block"""

    schema = "org.omf.v2.elements.blockmodel.sub"

    parent_block_count = properties.List(
        "Number of parent blocks along u, v, and w axes",
        properties.Integer("", min=1),
        min_length=3,
        max_length=3,
    )
    sub_block_count = properties.List(
        "Number of sub blocks in each parent block, along u, v, and w axes",
        properties.Integer("", min=1),
        min_length=3,
        max_length=3,
    )
    parent_block_size = properties.List(
        "Size of parent blocks in the u, v, and w dimensions",
        properties.Float("", min=0),
        min_length=3,
        max_length=3,
    )
    cbc = ArrayInstanceProperty(
        "Compressed block count - for regular sub block models this must "
        "have length equal to the product of parent_block_count and all "
        "values must be the product of sub_block_count (if attributes "
        "exist on the sub blocks), 1 (if attributes exist on the parent "
        "block) or 0; the default is an array of 1s",
        shape=("*",),
        dtype=(int, bool),
    )

    _valid_locations = ("parent_blocks", "sub_blocks")

    @properties.Array(
        "Compressed block index - used for indexing attributes "
        "into the sub block model; must have length equal to the "
        "product of parent_block_count plus 1 and monotonically increasing",
        shape=("*",),
        dtype=int,
        coerce=False,
    )
    def cbi(self):
        """Compressed block index"""
        if self.cbc is None:
            return None
        cbi = np.concatenate(
            [
                np.array([0], dtype=np.uint64),
                np.cumsum(self.cbc, dtype=np.uint64),
            ]
        )
        return cbi

    @properties.List(
        "Size of sub blocks in the u, v, and w dimensions",
        properties.Float("", min=0),
        min_length=3,
        max_length=3,
    )
    def sub_block_size(self):
        """Computed sub block size"""
        if not self.sub_block_count or not self.parent_block_size:
            return None
        return self.parent_block_size / np.array(self.sub_block_count)

    @properties.validator("parent_block_size")
    def _validate_size_is_not_zero(self, change):
        """Ensure block sizes are non-zero"""
        if 0 in change["value"]:
            raise properties.ValidationError(
                "Block size cannot be 0",
                prop="parent_block_size",
                instance=self,
                reason="invalid",
            )

    @properties.validator("cbc")
    def validate_cbc(self, change):
        """Ensure cbc is correct size and values"""
        value = change["value"]
        if not self.parent_block_count:
            pass
        elif len(value.array) != np.prod(self.parent_block_count):
            raise properties.ValidationError(
                "cbc must have length equal to the product of parent_block_count",
                prop="cbc",
                instance=self,
                reason="invalid",
            )
        if not self.sub_block_count:
            pass
        elif np.any(
            (value.array != 1)
            & (value.array != 0)
            & (value.array != np.prod(self.sub_block_count))
        ):
            raise properties.ValidationError(
                "cbc must have only values of prod(sub_block_count), 1, or 0",
                prop="cbc",
                instance=self,
                reason="invalid",
            )

    @property
    def num_cells(self):
        """Number of cells from last value in the compressed block index"""
        cbi = self.cbi
        if cbi is None:
            return None
        return cbi[-1]  # pylint: disable=E1136

    def location_length(self, location):
        """Return correct attribute length based on location"""
        if location == "parent_blocks":
            return np.sum(self.cbc.array.astype(np.bool))
        return self.num_cells

    def reset_cbc(self):
        """Reset cbc to no sub-blocks"""
        if not self.parent_block_count:
            raise ValueError("cannot reset cbc until parent_block_count is set")
        cbc_len = np.prod(self.parent_block_count)
        self.cbc = np.ones(cbc_len, dtype=np.uint32)

    def refine(self, ijk):
        """Refine parent blocks at a single ijk or a list of multiple ijks"""
        if self.cbc is None or not self.sub_block_count:
            raise ValueError(
                "Cannot refine sub block model without specifying number "
                "of parent and sub blocks"
            )
        try:
            inds = self.ijk_array_to_indices(ijk)
        except ValueError:
            inds = self.ijk_to_index(ijk)
        self.cbc.array[inds] = np.prod(self.sub_block_count)  # pylint: disable=E1137


class OctreeSubBlockModel(BaseBlockModel):
    """Block model where sub-blocks follow an octree pattern in each parent"""

    schema = "org.omf.v2.elements.blockmodel.octree"

    max_level = 8  # Maximum times blocks can be subdivided
    level_bits = 4  # Enough for 0 to 8 refinements

    parent_block_count = properties.List(
        "Number of parent blocks along u, v, and w axes",
        properties.Integer("", min=1),
        min_length=3,
        max_length=3,
    )
    parent_block_size = properties.List(
        "Size of parent blocks in the u, v, and w dimensions",
        properties.Float("", min=0),
        min_length=3,
        max_length=3,
    )
    cbc = ArrayInstanceProperty(
        "Compressed block count - for octree sub block models this must "
        "have length equal to the product of parent_block_count and each "
        "value must be equal to the number of octree sub blocks within "
        "the corresponding parent block (since max level is 8 in each "
        "dimension, the max number of sub blocks in a parent is (2^8)^3), "
        "1 (if parent block is not subdivided) or 0 (if parent block is "
        "unused); the default is an array of 1s",
        shape=("*",),
        dtype=(int, bool),
    )
    zoc = ArrayInstanceProperty(
        "Z-order curves - sub block location pointer and level, encoded as bits",
        shape=("*",),
        dtype=int,
    )

    _valid_locations = ("parent_blocks", "sub_blocks")

    @properties.Array(
        "Compressed block index - used for indexing attributes "
        "into the sub block model; must have length equal to the "
        "product of parent_block_count plus 1 and monotonically increasing",
        shape=("*",),
        dtype=int,
        coerce=False,
    )
    def cbi(self):
        """Compressed block index"""
        if self.cbc is None:
            return None
        cbi = np.concatenate(
            [
                np.array([0], dtype=np.uint64),
                np.cumsum(self.cbc, dtype=np.uint64),
            ]
        )
        return cbi

    @properties.validator("cbc")
    def validate_cbc(self, change):
        """Ensure cbc is correct size and values"""
        value = change["value"]
        if not self.parent_block_count:
            pass
        elif len(value.array) != np.prod(self.parent_block_count):
            raise properties.ValidationError(
                "cbc must have length equal to the product of parent_block_count",
                prop="cbc",
                instance=self,
                reason="invalid",
            )
        if np.max(value.array) > 8 ** 8 or np.min(value.array) < 0:
            raise properties.ValidationError(
                "cbc must have values between 0 and 8^8",
                prop="cbc",
                instance=self,
                reason="invalid",
            )

    @properties.validator("zoc")
    def validate_zoc(self, change):
        """Ensure Z-order curve array is correct length and valid values"""
        value = change["value"]
        cbi = self.cbi
        if cbi is None:
            pass
        elif len(value.array) != cbi[-1]:
            raise properties.ValidationError(
                "zoc must have length equal to maximum compressed block index value",
                prop="zoc",
                instance=self,
                reason="invalid",
            )
        max_curve_value = 268435448  # -> 0b1111111111111111111111111000
        if np.max(value.array) > max_curve_value or np.min(value.array) < 0:
            raise properties.ValidationError(
                "zoc must have values between 0 and 8^8",
                prop="cbc",
                instance=self,
                reason="invalid",
            )

    @property
    def num_cells(self):
        """Number of cells from last value in the compressed block index"""
        cbi = self.cbi
        if cbi is None:
            return None
        return cbi[-1]  # pylint: disable=E1136

    def location_length(self, location):
        """Return correct attribute length based on location"""
        if location == "parent_blocks":
            return np.sum(self.cbc.array.astype(np.bool))
        return self.num_cells

    def reset_cbc(self):
        """Reset cbc to no sub-blocks"""
        if not self.parent_block_count:
            raise ValueError("cannot reset cbc until parent_block_count is set")
        cbc_len = np.prod(self.parent_block_count)
        self.cbc = np.ones(cbc_len, dtype=np.uint32)

    def reset_zoc(self):
        """Reset zoc to no sub-blocks"""
        if not self.parent_block_count:
            raise ValueError("cannot reset zoc until parent_block_count is set")
        zoc_len = np.prod(self.parent_block_count)
        self.zoc = np.zeros(zoc_len, dtype=np.int32)

    @staticmethod
    def bitrange(index, width, start, end):
        """Extract a bit range as an integer

        [start, end) is inclusive lower bound, exclusive upper bound.
        """
        return index >> (width - end) & ((2 ** (end - start)) - 1)

    @classmethod
    def get_curve_value(cls, pointer, level):
        """Get Z-order curve value from pointer and level

        Values range from 0 (pointer=[0, 0, 0], level=0) to
        268435448 (pointer=[255, 255, 255], level=8).
        """
        idx = 0
        iwidth = cls.max_level * 3
        for i in range(iwidth):
            bitoff = cls.max_level - (i // 3) - 1
            poff = 3 - (i % 3) - 1
            bitrange = (
                cls.bitrange(
                    index=pointer[3 - 1 - poff],
                    width=cls.max_level,
                    start=bitoff,
                    end=bitoff + 1,
                )
                << i
            )
            idx |= bitrange
        return (idx << cls.level_bits) + level

    @classmethod
    def get_pointer(cls, curve_value):
        """Get pointer value from Z-order curve value

        Pointer values are length-3 with values between 0 and 255
        """
        index = curve_value >> cls.level_bits
        pointer = [0] * 3
        iwidth = cls.max_level * 3
        for i in range(iwidth):
            bitrange = (
                cls.bitrange(
                    index=index,
                    width=iwidth,
                    start=i,
                    end=i + 1,
                )
                << (iwidth - i - 1) // 3
            )
            pointer[i % 3] |= bitrange
        pointer.reverse()
        return pointer

    @classmethod
    def get_level(cls, curve_value):
        """Get level value from Z-order curve value

        Level comes from the last 4 bits, with values between 0 and 8
        """
        return curve_value & (2 ** cls.level_bits - 1)

    @classmethod
    def level_width(cls, level):
        """Width of a level, in bits

        Max level of 8 has level width of 1; min level of 0 has level
        width of 256.
        """
        if not 0 <= level <= cls.max_level:
            raise ValueError("level must be between 0 and {}".format(cls.max_level))
        return 2 ** (cls.max_level - level)

    def refine(self, index, ijk=None, refinements=1):
        """Subdivide at the given index

        .. note::
           This method is for demonstration only.
           It is impractical and not intended to build an octree blockmodel
           using this method alone.

        If ijk is provided, index is relative to ijk parent block.
        Otherwise, index is relative to the entire block model.

        By default, blocks are refined a single level, from 1 sub-block
        to 8 sub-blocks. However, a greater number of refinements may be
        specified, where the final number of sub-blocks equals
        (2**refinements)**3.
        """
        cbi = self.cbi
        if ijk is not None:
            index += int(cbi[self.ijk_to_index(ijk)])
        parent_index = np.sum(index >= cbi) - 1  # pylint: disable=W0143
        if not 0 <= index < len(self.zoc):
            raise ValueError("index must be between 0 and {}".format(len(self.zoc)))

        curve_value = self.zoc[index]
        level = self.get_level(curve_value)
        if not 0 <= refinements <= self.max_level - level:
            raise ValueError(
                "refinements must be between 0 and {}".format(self.max_level - level)
            )
        new_width = self.level_width(level + refinements)

        new_pointers = np.indices([2 ** refinements] * 3)
        new_pointers = new_pointers.reshape(3, (2 ** refinements) ** 3).T
        new_pointers = new_pointers * new_width

        pointer = self.get_pointer(curve_value)
        new_pointers = new_pointers + pointer

        new_curve_values = sorted(
            [
                self.get_curve_value(pointer, level + refinements)
                for pointer in new_pointers
            ]
        )

        self.cbc.array[parent_index] += len(new_curve_values) - 1
        self.zoc = np.concatenate(
            [
                self.zoc[:index],
                new_curve_values,
                self.zoc[index + 1 :],
            ]
        )


class ArbitrarySubBlockModel(BaseBlockModel):
    """Block model with arbitrary, variable sub-blocks"""

    parent_block_count = properties.List(
        "Number of parent blocks along u, v, and w axes",
        properties.Integer("", min=1),
        min_length=3,
        max_length=3,
    )
    parent_block_size = properties.List(
        "Size of parent blocks in the u, v, and w dimensions",
        properties.Float("", min=0),
        min_length=3,
        max_length=3,
    )
    cbc = ArrayInstanceProperty(
        "Compressed block count - for arbitrary sub block models this must "
        "have length equal to the product of parent_block_count and each "
        "value must be equal to the number of sub blocks within the "
        "corresponding parent block, 1 (if attributes exist on the parent "
        "block) or 0; the default is an array of 1s",
        shape=("*",),
        dtype=(int, bool),
    )
    sub_block_corners = ArrayInstanceProperty(
        "Block corners normalized 0-1 relative to parent block",
        shape=("*", 3),
        dtype=float,
    )
    sub_block_sizes = ArrayInstanceProperty(
        "Block widths normalized 0-1 relative to parent block",
        shape=("*", 3),
        dtype=float,
    )

    _valid_locations = ("parent_blocks", "sub_blocks")

    @properties.Array(
        "Compressed block index - used for indexing attributes "
        "into the sub block model; must have length equal to the "
        "product of parent_block_count plus 1 and monotonically increasing",
        shape=("*",),
        dtype=int,
        coerce=False,
    )
    def cbi(self):
        """Compressed block index"""
        if self.cbc is None:
            return None
        cbi = np.r_[
            np.array([0], dtype=np.uint64),
            np.cumsum(self.cbc, dtype=np.uint64),
        ]
        return cbi

    @properties.Array(
        "Block centroids normalized 0-1 relative to parent block",
        shape=("*", 3),
        dtype=float,
        coerce=False,
    )
    def sub_block_centroids(self):
        """Block centroids normalized 0-1 relative to parent block

        Computed from sub_block_corners and sub_block_sizes
        """
        if self.sub_block_corners is None or self.sub_block_sizes is None:
            return None
        return self.sub_block_corners.array + self.sub_block_sizes.array / 2

    @properties.Array(
        "Block corners relative to parent block",
        shape=("*", 3),
        dtype=float,
        coerce=False,
    )
    def sub_block_corners_absolute(self):
        """Block corners relative to parent block

        Computed from sub_block_corners and sub_block_sizes
        """
        if self.sub_block_corners is None or self.parent_block_size is None:
            return None
        cbc = self.cbc
        all_indices = np.array(range(len(cbc)), dtype=np.uint64)
        unique_parent_ijks = self.indices_to_ijk_array(all_indices)
        parent_ijks = np.repeat(unique_parent_ijks, cbc, axis=0)
        corners = parent_ijks + self.sub_block_corners
        return corners * self.parent_block_size

    @properties.Array(
        "Block centroids relative to parent block",
        shape=("*", 3),
        dtype=float,
        coerce=False,
    )
    def sub_block_centroids_absolute(self):
        """Block centroids relative to parent block

        Computed from sub_block_corners and sub_block_sizes
        """
        if self.sub_block_centroids is None or self.parent_block_size is None:
            return None
        cbc = self.cbc
        all_indices = np.array(range(len(cbc)), dtype=np.uint64)
        unique_parent_ijks = self.indices_to_ijk_array(all_indices)
        parent_ijks = np.repeat(unique_parent_ijks, cbc, axis=0)
        centroids = parent_ijks + self.sub_block_centroids
        return centroids * self.parent_block_size

    @properties.Array(
        "Block widths relative to parent block",
        shape=("*", 3),
        dtype=float,
        coerce=False,
    )
    def sub_block_sizes_absolute(self):
        """Block widths relative to parent block

        Computed from sub_block_corners and sub_block_sizes
        """
        if self.sub_block_sizes is None or self.parent_block_size is None:
            return None
        return self.sub_block_sizes.array * self.parent_block_size

    @properties.validator("parent_block_size")
    def _validate_size_is_not_zero(self, change):
        """Ensure parent blocks are non-zero"""
        if 0 in change["value"]:
            raise properties.ValidationError(
                "Block size cannot be 0",
                prop="parent_block_size",
                instance=self,
                reason="invalid",
            )

    @properties.validator("cbc")
    def validate_cbc(self, change):
        """Ensure cbc is correct size and values"""
        value = change["value"]
        if not self.parent_block_count:
            pass
        elif len(value.array) != np.prod(self.parent_block_count):
            raise properties.ValidationError(
                "cbc must have length equal to the product of parent_block_count",
                prop="cbc",
                instance=self,
                reason="invalid",
            )
        if np.min(value.array) < 0:
            raise properties.ValidationError(
                "cbc values must be non-negative",
                prop="cbc",
                instance=self,
                reason="invalid",
            )
        return value

    def validate_sub_block_attributes(self, value, prop_name):
        """Ensure value is correct length"""
        cbi = self.cbi
        if cbi is None:
            return value
        if len(value) != cbi[-1]:
            raise properties.ValidationError(
                "{} attributes must have length equal to "
                "total number of sub blocks".format(prop_name),
                prop=prop_name,
                instance=self,
                reason="invalid",
            )
        return value

    @properties.validator("sub_block_corners")
    def _validate_sub_block_corners(self, change):
        """Validate sub block corners array is correct length"""
        change["value"] = self.validate_sub_block_attributes(
            change["value"], "sub_block_corners"
        )

    @properties.validator("sub_block_sizes")
    def _validate_sub_block_sizes(self, change):
        """Validate sub block size array is correct length and positive"""
        value = self.validate_sub_block_attributes(change["value"], "sub_block_sizes")
        if np.min(value.array) <= 0:
            raise properties.ValidationError(
                "sub block sizes must be positive",
                prop="sub_block_sizes",
                instance=self,
                reason="invalid",
            )

    @property
    def num_cells(self):
        """Number of cells from last value in the compressed block index"""
        cbi = self.cbi
        if cbi is None:
            return None
        return cbi[-1]  # pylint: disable=E1136

    def location_length(self, location):
        """Return correct attribute length based on location"""
        if location == "parent_blocks":
            return np.sum(self.cbc.array.astype(np.bool))
        return self.num_cells

    def reset_cbc(self):
        """Reset cbc to no sub-blocks"""
        if not self.parent_block_count:
            raise ValueError("cannot reset cbc until parent_block_count is set")
        cbc_len = np.prod(self.parent_block_count)
        self.cbc = np.ones(cbc_len, dtype=np.uint32)
