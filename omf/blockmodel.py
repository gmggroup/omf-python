"""blockmodel.py: Block Model element definitions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties

from .base import ProjectElement


class BaseBlockModel(ProjectElement):
    """Basic orientation properties for all block models"""
    axis_u = properties.Vector3(
        'Vector orientation of u-direction',
        default='X',
        length=1,
    )
    axis_v = properties.Vector3(
        'Vector orientation of v-direction',
        default='Y',
        length=1,
    )
    axis_w = properties.Vector3(
        'Vector orientation of w-direction',
        default='Z',
        length=1,
    )
    corner = properties.Vector3(
        'Corner of the block model relative to Project coordinate reference system',
        default=[0., 0., 0.],
    )

    @properties.validator
    def _validate_axes(self):
        """Check if mesh content is built correctly"""
        if not (np.abs(self.axis_u.dot(self.axis_v) < 1e-6) and                #pylint: disable=no-member
                np.abs(self.axis_v.dot(self.axis_w) < 1e-6) and                #pylint: disable=no-member
                np.abs(self.axis_w.dot(self.axis_u) < 1e-6)):                  #pylint: disable=no-member
            raise ValueError('axis_u, axis_v, and axis_w must be orthogonal')
        return True

    @property
    def num_parent_blocks(self):
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
        blocks = self.num_parent_blocks
        if not blocks:
            raise AttributeError(
                'num_parent_blocks is required to calculate index'
            )
        if not isinstance(ijk_array, (list, tuple, np.ndarray)):
            raise ValueError('ijk_array must be a list of length-3 ijk values')
        ijk_array = np.array(ijk_array)
        if len(ijk_array.shape) != 2 or ijk_array.shape[1] != 3:
            raise ValueError('ijk_array must be n x 3 array')
        if not np.array_equal(ijk_array, ijk_array.astype(np.uint32)):
            raise ValueError('ijk values must be integers')
        if np.any(np.min(ijk_array, axis=0) >= self.num_parent_blocks):
            raise ValueError(
                'ijk must be less than num_parent_blocks in each dimension'
            )
        index = np.ravel_multi_index(
            multi_index=ijk_array.T,
            dims=self.num_parent_blocks,
            order='F',
        )
        return index


class TensorBlockModel(BaseBlockModel):
    """Block model with variable spacing in each dimension"""
    schema_type = 'org.omf.v2.element.blockmodel.tensor'

    tensor_u = properties.Array(
        'Tensor cell widths, u-direction',
        shape=('*',),
        dtype=float,
    )
    tensor_v = properties.Array(
        'Tensor cell widths, v-direction',
        shape=('*',),
        dtype=float,
    )
    tensor_w = properties.Array(
        'Tensor cell widths, w-direction',
        shape=('*',),
        dtype=float,
    )

    _valid_locations = ('vertices', 'cells')

    def location_length(self, location):
        """Return correct data length based on location"""
        if location == 'cells':
            return self.num_cells
        return self.num_nodes

    def _tensors_defined(self):
        tensors = [self.tensor_u, self.tensor_v, self.tensor_w]
        return all((tensor is not None for tensor in tensors))

    @property
    def num_nodes(self):
        """Number of nodes (vertices)"""
        if not self._tensors_defined():
            return None
        nodes = (
            (len(self.tensor_u)+1) *
            (len(self.tensor_v)+1) *
            (len(self.tensor_w)+1)
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
    def num_parent_blocks(self):
        if not self._tensors_defined:
            return None
        blocks = [len(self.tensor_u), len(self.tensor_v), len(self.tensor_w)]
        return blocks


class RegularBlockModel(BaseBlockModel):
    """Block model with constant spacing in each dimension"""

    schema_type = 'org.omf.v2.elements.blockmodel.regular'

    num_blocks = properties.List(
        'Number of blocks along u, v, and w axes',
        properties.Integer('', min=1),
        min_length=3,
        max_length=3,
    )
    size_blocks = properties.List(
        'Size of blocks in the u, v, and w dimensions',
        properties.Float('', min=0),
        min_length=3,
        max_length=3,
    )

    @properties.validator('size_blocks')
    def _validate_size_is_not_zero(self, change):
        if 0 in change['value']:
            raise properties.ValidationError(
                'Block size cannot be 0',
                prop='size_blocks',
                instance=self,
                reason='invalid',
            )

    _valid_locations = ('cells',)

    @properties.Array(
        'Compressed block count - for regular block models this must '
        'have length equal to the product of num_blocks and all values '
        'must be 1 (if attributes exist on the block) or 0; the default '
        'is an array of 1s',
        shape=('*',),
        dtype=(int, bool),
        coerce=False,
    )
    def cbc(self):
        """Compressed block count"""
        cbc_cache = getattr(self, '_cbc', None)
        if not self.num_blocks:
            return cbc_cache
        cbc_len = np.prod(self.num_blocks)
        if cbc_cache is None or len(cbc_cache) != cbc_len:
            self._cbc = np.ones(cbc_len, dtype=np.bool)                        # pylint: disable=attribute-defined-outside-init
        return self._cbc

    @cbc.setter
    def cbc(self, value):
        self._cbc = self.validate_cbc(value)                                   # pylint: disable=attribute-defined-outside-init

    def validate_cbc(self, value):
        """Ensure cbc is correct size and values"""
        if self.num_blocks and len(value) != np.prod(self.num_blocks):
            raise properties.ValidationError(
                'cbc must have length equal to the product '
                'of num_blocks',
                prop='cbc',
                instance=self,
                reason='invalid',
            )
        if np.max(value) > 1 or np.min(value) < 0:
            raise properties.ValidationError(
                'cbc must have only values 0 or 1',
                prop='cbc',
                instance=self,
                reason='invalid',
            )
        return value

    @properties.validator
    def _validate_cbc(self):
        self.validate_cbc(self.cbc)


    @properties.Array(
        'Compressed block index - used for indexing attributes '
        'into the block model; must have length equal to the '
        'product of num_blocks plus 1 and monotonically increasing',
        shape=('*',),
        dtype=int,
        coerce=False,
    )
    def cbi(self):
        """Compressed block index"""
        if self.cbc is None:
            return None
        # Recalculating the sum on the fly is faster than checking md5
        cbi = np.r_[
            np.array([0], dtype=np.uint32),
            np.cumsum(self.cbc, dtype=np.uint32),
        ]
        return cbi

    @property
    def num_cells(self):
        """Number of cells from last value in the compressed block index"""
        if self.cbi is None:
            return None
        return self.cbi[-1]                                                    # pylint: disable=unsubscriptable-object

    def location_length(self, location):
        """Return correct data length based on location"""
        return self.num_cells

    @property
    def num_parent_blocks(self):
        return self.num_blocks


class RegularSubBlockModel(BaseBlockModel):
    """Regular block model with an additional level of sub-blocks"""

    schema_type = 'org.omf.v2.elements.blockmodel.sub'

    num_parent_blocks = properties.List(
        'Number of parent blocks along u, v, and w axes',
        properties.Integer('', min=1),
        min_length=3,
        max_length=3,
    )
    num_sub_blocks = properties.List(
        'Number of sub blocks in each parent block, along u, v, and w axes',
        properties.Integer('', min=1),
        min_length=3,
        max_length=3,
    )
    size_parent_blocks = properties.List(
        'Size of parent blocks in the u, v, and w dimensions',
        properties.Float('', min=0),
        min_length=3,
        max_length=3,
    )

    @properties.List(
        'Size of sub blocks in the u, v, and w dimensions',
        properties.Float('', min=0),
        min_length=3,
        max_length=3,
    )
    def size_sub_blocks(self):
        """Computed sub block size"""
        if not self.num_sub_blocks or not self.size_parent_blocks:
            return None
        return self.size_parent_blocks / np.array(self.num_sub_blocks)

    @properties.validator('size_parent_blocks')
    def _validate_size_is_not_zero(self, change):
        if 0 in change['value']:
            raise properties.ValidationError(
                'Block size cannot be 0',
                prop='size_blocks',
                instance=self,
                reason='invalid',
            )

    _valid_locations = ('parent_blocks', 'sub_blocks')

    @properties.Array(
        'Compressed block count - for regular sub block models this must '
        'have length equal to the product of num_parent_blocks and all '
        'values must be the product of num_sub_blocks (if attributes '
        'exist on the sub blocks), 1 (if attributes exist on the parent '
        'block) or 0; the default is an array of 1s',
        shape=('*',),
        dtype=(int, bool),
        coerce=False,
    )
    def cbc(self):
        """Compressed block count"""
        cbc_cache = getattr(self, '_cbc', None)
        if not self.num_parent_blocks or not self.num_sub_blocks:
            return cbc_cache
        cbc_len = np.prod(self.num_parent_blocks)
        if cbc_cache is None or len(cbc_cache) != cbc_len:
            self._cbc = np.ones(cbc_len, dtype=np.uint32)                        # pylint: disable=attribute-defined-outside-init
        return self._cbc

    @cbc.setter
    def cbc(self, value):
        self._cbc = self.validate_cbc(value)                                   # pylint: disable=attribute-defined-outside-init

    def validate_cbc(self, value):
        """Ensure cbc is correct size and values"""
        if not self.num_parent_blocks:
            pass
        elif len(value) != np.prod(self.num_parent_blocks):
            raise properties.ValidationError(
                'cbc must have length equal to the product '
                'of num_parent_blocks',
                prop='cbc',
                instance=self,
                reason='invalid',
            )
        if not self.num_sub_blocks:
            pass
        elif np.any(
                (value != 1) & (value != 0) &
                (value != np.prod(self.num_sub_blocks))
        ):
            raise properties.ValidationError(
                'cbc must have only values of prod(num_sub_blocks), 1, or 0',
                prop='cbc',
                instance=self,
                reason='invalid',
            )
        return value

    @properties.validator
    def _validate_cbc(self):
        self.validate_cbc(self.cbc)

    @properties.Array(
        'Compressed block index - used for indexing attributes '
        'into the sub block model; must have length equal to the '
        'product of num_parent_blocks plus 1 and monotonically increasing',
        shape=('*',),
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

    @property
    def num_cells(self):
        """Number of cells from last value in the compressed block index"""
        if self.cbi is None:
            return None
        return self.cbi[-1]                                                    # pylint: disable=unsubscriptable-object

    def location_length(self, location):
        """Return correct data length based on location"""
        if location == 'parent_blocks':
            return np.sum(self.cbc.astype(np.bool))                            # pylint: disable=no-member
        return self.num_cells

    def refine(self, ijk):
        """Refine parent blocks at a single ijk or a list of multiple ijks"""
        if self.cbc is None or not self.num_sub_blocks:
            raise ValueError(
                'Cannot refine sub block model without specifying number '
                'of parent and sub blocks'
            )
        try:
            inds = self.ijk_array_to_indices(ijk)
        except ValueError:
            inds = self.ijk_to_index(ijk)
        self.cbc[inds] = np.prod(self.num_sub_blocks)                          # pylint: disable=unsupported-assignment-operation


class OctreeSubBlockModel(BaseBlockModel):
    """Block model where sub-blocks follow an octree pattern"""

    schema_type = 'org.omf.v2.elements.blockmodel.octree'

    max_level = 8  # Maximum times blocks can be subdivided
    level_bits = 4  # Enough for 0 to 8 refinements

    num_parent_blocks = properties.List(
        'Number of parent blocks along u, v, and w axes',
        properties.Integer('', min=1),
        min_length=3,
        max_length=3,
    )
    size_parent_blocks = properties.List(
        'Size of parent blocks in the u, v, and w dimensions',
        properties.Float('', min=0),
        min_length=3,
        max_length=3,
    )

    _valid_locations = ('parent_blocks', 'sub_blocks')

    @properties.Array(
        'Compressed block count - for octree sub block models this must '
        'have length equal to the product of num_parent_blocks and each '
        'value must be equal to the number of octree sub blocks within '
        'the corresponding parent block (since max level is 8 in each '
        'dimension, the max number of sub blocks in a parent is (2^8)^3), '
        '1 (if parent block is not subdivided) or 0 (if parent block is '
        'unused); the default is an array of 1s',
        shape=('*',),
        dtype=(int, bool),
        coerce=False,
    )
    def cbc(self):
        """Compressed block count"""
        cbc_cache = getattr(self, '_cbc', None)
        if not self.num_parent_blocks:
            return cbc_cache
        cbc_len = np.prod(self.num_parent_blocks)
        if cbc_cache is None or len(cbc_cache) != cbc_len:
            self._cbc = np.ones(cbc_len, dtype=np.uint32)                        # pylint: disable=attribute-defined-outside-init
        return self._cbc

    @cbc.setter
    def cbc(self, value):
        self._cbc = self.validate_cbc(value)                                   # pylint: disable=attribute-defined-outside-init

    def validate_cbc(self, value):
        """Ensure cbc is correct size and values"""
        if not self.num_parent_blocks:
            pass
        elif len(value) != np.prod(self.num_parent_blocks):
            raise properties.ValidationError(
                'cbc must have length equal to the product '
                'of num_parent_blocks',
                prop='cbc',
                instance=self,
                reason='invalid',
            )
        if np.max(value) > 8**8 or np.min(value) < 0:
            raise properties.ValidationError(
                'cbc must have values between 0 and 8^8',
                prop='cbc',
                instance=self,
                reason='invalid',
            )
        return value

    @properties.validator
    def _validate_cbc(self):
        self.validate_cbc(self.cbc)

    @properties.Array(
        'Compressed block index - used for indexing attributes '
        'into the sub block model; must have length equal to the '
        'product of num_parent_blocks plus 1 and monotonically increasing',
        shape=('*',),
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
        'Z-order curves - sub block location pointer and level, encoded '
        'as bits',
        shape=('*',),
        dtype=int,
        coerce=False,
    )
    def zoc(self):
        zoc_cache = getattr(self, '_zoc', None)
        if not self.num_parent_blocks:
            return zoc_cache
        if zoc_cache is None:
            self._zoc = np.zeros(
                np.prod(self.num_parent_blocks),
                dtype=np.int32,
            )
        return self._zoc

    @zoc.setter
    def zoc(self, value):
        self._zoc = self.validate_zoc(value)

    def validate_zoc(self, value):
        """Ensure Z-order curve array is correct length and valid values"""
        cbi = self.cbi
        if cbi is None:
            pass
        elif len(value) != cbi[-1]:
            raise properties.ValidationError(
                'zoc must have length equal to maximum compressed block '
                'index value',
                prop='zoc',
                instance=self,
                reason='invalid',
            )
        max_curve_value = 268435448  # -> 0b1111111111111111111111111000
        if np.max(value) > max_curve_value or np.min(value) < 0:
            raise properties.ValidationError(
                'zoc must have values between 0 and 8^8',
                prop='cbc',
                instance=self,
                reason='invalid',
            )
        return value

    @properties.validator
    def _validate_zoc(self):
        self.validate_zoc(self.zoc)

    @property
    def num_cells(self):
        """Number of cells from last value in the compressed block index"""
        if self.cbi is None:
            return None
        return self.cbi[-1]                                                    # pylint: disable=unsubscriptable-object

    def location_length(self, location):
        """Return correct data length based on location"""
        if location == 'parent_blocks':
            return np.sum(self.cbc.astype(np.bool))                            # pylint: disable=no-member
        return self.num_cells

    @staticmethod
    def bitrange(x, width, start, end):
        """Extract a bit range as an integer

        [start, end) is inclusive lower bound, exclusive upper bound.
        """
        return x >> (width - end) & ((2 ** (end - start)) - 1)

    @classmethod
    def get_curve_value(self, pointer, level):
        """Get Z-order curve value from pointer and level

        Values range from 0 (pointer=[0, 0, 0], level=0) to
        268435448 (pointer=[255, 255, 255], level=8).
        """
        idx = 0
        iwidth = self.max_level * 3
        for i in range(iwidth):
            bitoff = self.max_level - (i // 3) - 1
            poff = 3 - (i % 3) - 1
            b = self.bitrange(
                x=pointer[3 - 1 - poff],
                width=self.max_level,
                start=bitoff,
                end=bitoff + 1,
            ) << i
            idx |= b
        return (idx << self.level_bits) + level

    @classmethod
    def get_pointer(self, curve_value):
        """Get pointer value from Z-order curve value

        Pointer values are length-3 with values between 0 and 255
        """
        index = curve_value >> self.level_bits
        pointer = [0] * 3
        iwidth = self.max_level * 3
        for i in range(iwidth):
            b = self.bitrange(
                x=index,
                width=iwidth,
                start=i,
                end=i + 1,
            ) << (iwidth - i - 1) // 3
            pointer[i % 3] |= b
        pointer.reverse()
        return pointer

    @classmethod
    def get_level(self, curve_value):
        """Get level value from Z-order curve value

        Level comes from the last 4 bits, with values between 0 and 8
        """
        return curve_value & (2 ** self.level_bits - 1)

    @classmethod
    def level_width(self, level):
        """Width of a level, in bits

        Max level of 8 has level width of 1; min level of 0 has level
        width of 256.
        """
        if not 0 <= level <= self.max_level:
            raise ValueError(
                'level must be between 0 and {}'.format(max_level)
            )
        return 2 ** (self.max_level - level)

    def refine(self, index, ijk=None, refinements=1):
        """Subdivide at the given index

        If ijk is provided, index is relative to ijk parent block.
        Otherwise, index is relative to the entire block model.

        By default, blocks are refined a single level, from 1 sub-block
        to 8 sub-blocks. However, a greater number of refinements may be
        specified, where the final number of sub-blocks equals
        (2**refinements)**3.
        """
        if ijk is not None:
            index += int(self.cbi[self.ijk_to_index(ijk)])
        parent_index = np.sum(index >= self.cbi) - 1
        if not 0 <= index < len(self.zoc):
            raise ValueError(
                'index must be between 0 and {}'.format(len(self.zoc))
            )

        curve_value = self.zoc[index]
        level = self.get_level(curve_value)
        if not 0 <= refinements <= self.max_level - level:
            raise ValueError(
                'refinements must be between 0 and {}'.format(
                    self.max_level - level
                )
            )
        new_width = self.level_width(level + refinements)

        new_pointers = np.indices([2**refinements]*3)
        new_pointers = new_pointers.reshape(3, (2**refinements)**3).T
        new_pointers = new_pointers * new_width

        pointer = self.get_pointer(curve_value)
        new_pointers = new_pointers + pointer

        new_curve_values = sorted([
            self.get_curve_value(pointer, level + refinements)
            for pointer in new_pointers
        ])

        self.cbc[parent_index] += len(new_curve_values) - 1
        self.zoc = np.r_[
            self.zoc[:index],
            new_curve_values,
            self.zoc[index+1:],
        ]
