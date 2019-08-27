"""texture.py: contains ImageTexture definition"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties

from .base import ContentModel
from .data import Vector2Array
from .serializers import png_serializer, png_deserializer


class ImageTexture(ContentModel):
    """Contains an image that can be mapped to a point set or surface"""
    origin = properties.Vector3(
        'Origin point of the texture',
        default=[0., 0., 0.],
    )
    axis_u = properties.Vector3(
        'Vector corresponding to the image x-axis',
        default='X',
    )
    axis_v = properties.Vector3(
        'Vector corresponding to the image y-axis',
        default='Y',
    )
    image = properties.ImagePNG(
        'PNG image file',
        serializer=png_serializer,
        deserializer=png_deserializer,
    )


class UVTexture(ContentModel):
    """Contains an image that is UV mapped to a geometry"""

    image = properties.ImagePNG(
        'PNG image file',
        serializer=png_serializer,
        deserializer=png_deserializer,
    )
    uv_coordinates = properties.Instance(
        'Normalized UV coordinates mapping the image element vertices',
        Vector2Array,
    )
    @properties.validator
    def _validate_uv(self):
        """Validate UV values between 0 and 1"""
        if np.min(self.uv_coordinates) < 0 or np.max(self.uv_coordinates) > 1:
            raise properties.ValidationError(
                'UV coordinates must be between 0 and 1'
            )
        return True


class HasTexturesMixin(properties.HasProperties):
    """Mixin for elements with textures"""

    textures = properties.List(
        'Images mapped on the element',
        prop=properties.Union('', (ImageTexture, UVTexture)),
        required=False,
        default=list,
    )

    @properties.validator
    def _validate_textures(self):
        """Validate UVTextures against geometry"""
        if not hasattr(self, 'num_nodes'):
            return True
        for i, tex in enumerate(self.textures):
            if isinstance(tex, ImageTexture):
                continue
            if len(tex.uv_coordinates) != self.num_nodes:
                raise properties.ValidationError(
                    'data[{index}] length {datalen} does not match '
                    'vertices length {meshlen}'.format(
                        index=i,
                        datalen=len(tex.uv_coordinates),
                        meshlen=self.num_nodes,
                    )
                )
        return True
