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
        raise NotImplementedError()

    def ijk_to_index(self, ijk):
        return self.ijk_array_to_indices([ijk])[0]

    def ijk_array_to_indices(self, ijk_array):
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
        return all([tensor is not None for tensor in tensors])

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
        if location == 'parent_blocks':
            return np.sum(self.cbc.astype(np.bool))
        return self.num_cells

    def refine(self, ijk):
        try:
            inds = self.ijk_array_to_indices(ijk)
        except ValueError:
            inds = self.ijk_to_index(ijk)
        self.cbc[inds] = np.prod(self.num_sub_blocks)
