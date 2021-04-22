"""texture.py: contains ImageTexture definition"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties

from .base import ContentModel
from .data import Vector2Array
from .serializers import png_serializer, png_deserializer


class ProjectedTexture(ContentModel):
    """Contains an image that can be projected onto a point set or surface"""
    class_type = 'org.omf.v2.texture.projected'

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


class UVMappedTexture(ContentModel):
    """Contains an image that is UV mapped to a geometry"""

    image = properties.ImagePNG(
        'PNG image file',
        serializer=png_serializer,
        deserializer=png_deserializer,
    )
    uv_coordinates = properties.Instance(
        'Normalized UV coordinates mapping the image to element vertices; '
        'for values outside 0-1 the texture repeats at every integer level, '
        'and NaN indicates no texture at a vertex',
        Vector2Array,
    )


class HasTexturesMixin(properties.HasProperties):
    """Mixin for elements with textures"""

    textures = properties.List(
        'Images mapped on the element',
        prop=properties.Union('', (ProjectedTexture, UVMappedTexture)),
        required=False,
        default=list,
    )

    @properties.validator
    def _validate_textures(self):
        """Validate UVTextures against geometry"""
        if not hasattr(self, 'num_nodes'):
            return True
        for i, tex in enumerate(self.textures):
            if isinstance(tex, ProjectedTexture):
                continue
            if len(tex.uv_coordinates) != self.num_nodes:
                raise properties.ValidationError(
                    'texture[{index}] length {datalen} does not match '
                    'vertices length {meshlen}'.format(
                        index=i,
                        datalen=len(tex.uv_coordinates),
                        meshlen=self.num_nodes,
                    )
                )
        return True
