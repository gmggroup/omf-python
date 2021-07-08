"""texture.py: contains ImageTexture definition"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import uuid

import properties

from .base import BaseModel, ContentModel
from .attribute import ArrayInstanceProperty


class Image(BaseModel):
    """Class for setting PNG image"""
    schema = 'org.omf.v2.image.png'

    image = properties.ImagePNG(
        'PNG image file',
        serializer=lambda *args, **kwargs: None,
        deserializer=lambda *args, **kwargs: None,
    )

    def __init__(self, image=None, **kwargs):
        super(Image, self).__init__(**kwargs)
        if image is not None:
            self.image = image

    @properties.StringChoice(
        'Image data type string', choices=['png']
    )                                                                          #pylint: disable=no-self-use
    def data_type(self):
        """Image type descriptor, currently only PNGs are supported"""
        return 'png'

    @properties.Integer('Size of image in bytes')
    def size(self):
        """Total size of the array in bits"""
        if self.image is None:
            return None
        size = self.image.seek(0, 2)
        self.image.seek(0)
        return size

    def serialize(self, include_class=True, save_dynamic=False, **kwargs):
        output = super(Image, self).serialize(
            include_class=include_class, save_dynamic=True, **kwargs
        )
        image_uid = str(uuid.uuid4())
        binary_dict = kwargs.get('binary_dict', None)
        if binary_dict is not None:
            self.image.seek(0)
            binary_dict.update({image_uid: self.image.read()})
        output.update({'image': image_uid})
        return output

    @classmethod
    def deserialize(cls, value, trusted=False, strict=False,
                    assert_valid=False, **kwargs):
        binary_dict = kwargs.get('binary_dict', {})
        if not isinstance(value, dict):
            pass
        elif 'image' not in value:
            pass
        elif value['image'] in binary_dict:
            return cls(io.BytesIO(binary_dict[value['image']]))
        return cls()

class ProjectedTexture(ContentModel):
    """Contains an image that can be projected onto a point set or surface"""
    schema = 'org.omf.v2.texture.projected'

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
    image = properties.Instance(
        'PNG image file',
        Image,
    )


class UVMappedTexture(ContentModel):
    """Contains an image that is UV mapped to a geometry"""

    image = properties.Instance(
        'PNG image file',
        Image,
    )
    uv_coordinates = ArrayInstanceProperty(
        'Normalized UV coordinates mapping the image to element vertices; '
        'for values outside 0-1 the texture repeats at every integer level, '
        'and NaN indicates no texture at a vertex',
        shape=('*', 2),
        dtype=float,
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
            if len(tex.uv_coordinates.array) != self.num_nodes:
                raise properties.ValidationError(
                    'texture[{index}] length {datalen} does not match '
                    'vertices length {meshlen}'.format(
                        index=i,
                        datalen=len(tex.uv_coordinates.array),
                        meshlen=self.num_nodes,
                    )
                )
        return True
