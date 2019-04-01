"""blockmodel.py: Block Model element definitions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties

from .base import ProjectElement


class RegularBlockModel(ProjectElement):
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

    _valid_locations = ('cells',)

    @properties.Property(
        'Compressed block count - for regular block models this must '
        'have length equal to the product of num_blocks and all values '
        'must be 1 (if attributes exist on the block) or 0; the default '
        'is an array of 1s',
    )
    def cbc(self):
        cbc_cache = getattr(self, '_cbc', None)
        if not self.num_blocks:
            return cbc_cache
        cbc_len = np.prod(self.num_blocks)
        if cbc_cache is None or len(cbc_cache) != cbc_len:
            self._cbc = np.ones(cbc_len, dtype='int8')
        return self._cbc

    @cbc.setter
    def cbc(self, value):
        self._cbc = self.validate_cbc(value)

    def validate_cbc(self, value):
        """Ensure cbc is correct size and values"""
        if isinstance(value, (list, tuple)):
            value = np.array(value, dtype='int8')
        if not isinstance(value, np.ndarray):
            raise properties.ValidationError(
                'cbc must be an array',
                prop='cbc',
                instance=self,
                reason='invalid',
            )
        if len(value.shape) != 1:
            raise properties.ValidationError(
                'cbc must be a 1-dimensional array',
                prop='cbc',
                instance=self,
                reason='invalid',
            )
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
    )
    def cbi(self):
        if self.cbc is None:
            return
        # Recalculating the sum on the fly is faster than checking md5
        cbi = np.r_[
            np.array([0], dtype='int8'),
            np.cumsum(self.cbc, dtype='int8'),
        ]
        return cbi

    @property
    def num_cells(self):
        return self.cbi[-1]
