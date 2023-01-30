import io
import json
import numpy as np
import struct
import uuid
import zlib

from .interface import IOMFReader, InvalidOMFFile

from .. import attribute, base, lineset, pointset, texture

COMPATIBILITY_VERSION = b'OMF-v0.9.0'
_default = object()


class Reader(IOMFReader):
    def __init__(self, filename: str):
        self._filename = filename
        self._f = None
        self._project = None
        self._include_binary = True
        self._attribute_bucket = dict()

    def load(self, include_binary: bool = True, project_json: bool = None):
        self._include_binary = include_binary

        with open(self._filename, 'rb') as self._f:
            project_uuid, json_start = self._read_header()
            self._project = self._read_json(json_start)
            return self._copy_project(project_uuid)

    def _test_data_needed(self, *args, **kwargs):
        # temporary placeholder
        raise InvalidOMFFile('Test data required')

    def _read_header(self):
        """Checks magic number and version; gets project uid and json start"""
        self._f.seek(0)
        if self._f.read(4) != b'\x84\x83\x82\x81':
            raise InvalidOMFFile(f'Unsupported format: {self._filename}')
        file_version = struct.unpack('<32s', self._f.read(32))[0]
        file_version = file_version[0:len(COMPATIBILITY_VERSION)]
        if file_version != COMPATIBILITY_VERSION:
            raise InvalidOMFFile("Unsupported file version: {}".format(file_version))
        project_uuid = uuid.UUID(bytes=struct.unpack('<16s', self._f.read(16))[0])
        json_start = struct.unpack('<Q', self._f.read(8))[0]
        return str(project_uuid), json_start

    def _read_json(self, json_start):
        self._f.seek(json_start, 0)
        return json.loads(self._f.read().decode('utf-8'))

    # Safe access to attributes
    @classmethod
    def __get_attr(cls, src, attr, optional=False, converter=None, default=_default):
        if attr not in src:
            if not optional:
                raise InvalidOMFFile(f"Attribute {attr} missing")
            if default is _default:
                return None
            value = default
        else:
            value = src[attr]
            if converter:
                value = converter(value)
        return value

    @classmethod
    def __require_attr(cls, src, attr, required_value):
        value = cls.__get_attr(src, attr)
        if value != required_value:
            raise InvalidOMFFile(f"Invalid attribute {attr}. Expected: {required_value}, actual: {value}")

    @classmethod
    def __copy_attr(cls, src, src_attr, dst, dst_attr=None,
                    optional_src=False, optional_dst=False, converter=None, default=_default):
        if dst_attr is None:
            dst_attr = src_attr
        if src_attr not in src:
            if not optional_src:
                raise InvalidOMFFile(f"Attribute {src_attr} missing")
            if default is _default:
                return
            value = default
        else:
            value = src[src_attr]
            if converter:
                value = converter(value)

        if optional_dst and default is not _default and value == default:
            return

        if isinstance(dst, dict):
            dst[dst_attr] = value
        else:
            setattr(dst, dst_attr, value)

    # reading arrays
    def _load_array(self, scalar_array):
        scalar_class = self.__get_attr(scalar_array, '__class__')
        converter_lookup = {
            'StringArray': self._test_data_needed,
            'DateTimeArray': self._test_data_needed,
            'ColorArray': self._test_data_needed,
        }
        shape_lookup = {
            'ScalarArray': ('*',),
            'Int2Array': ('*', 2),
            'Int3Array': ('*', 3),
            'Vector2Array': ('*', 2),
            'Vector3Array': ('*', 3),
        }
        converter = self.__get_attr(converter_lookup, scalar_class, optional=True)
        if converter is not None:
            return converter(scalar_array)

        shape = self.__get_attr(shape_lookup, scalar_class)
        shape = tuple(-1 if s == '*' else s for s in shape)
        base_vector = self.__get_attr(scalar_array, 'array')

        start = self.__get_attr(base_vector, 'start')
        length = self.__get_attr(base_vector, 'length')
        dtype = self.__get_attr(base_vector, 'dtype')

        self._f.seek(start)
        buffer = zlib.decompress(self._f.read(length))
        return np.frombuffer(buffer, dtype).reshape(shape)

    def _copy_scalar_array(self, src, src_attr, dst, dst_attr=None):
        if not self._include_binary:
            return
        if dst_attr is None:
            dst_attr = src_attr
        value_uid = self.__get_attr(src, src_attr)
        scalar_array = self.__get_attr(self._project, value_uid)
        array = self._load_array(scalar_array)
        setattr(dst, dst_attr, array)

    def _load_image(self, image_png):
        start = self.__get_attr(image_png, 'start')
        length = self.__get_attr(image_png, 'length')
        self.__require_attr(image_png, 'dtype', 'image/png')
        self._f.seek(start)
        img = io.BytesIO()
        img.write(zlib.decompress(self._f.read(length)))
        img.seek(0, 0)
        return texture.Image(img)

    def _copy_image_png(self, src, src_attr, dst, dst_attr=None):
        if not self._include_binary:
            return
        if dst_attr is None:
            dst_attr = src_attr
        image_png = self.__get_attr(src, src_attr)
        image = self._load_image(image_png)
        setattr(dst, dst_attr, image)

    # base-class handlers
    def _copy_uid_model(self, src, dst):
        self.__copy_attr(src, 'date_created', dst.metadata)
        self.__copy_attr(src, 'date_modified', dst.metadata)

    def _copy_content_model(self, src, dst):
        self._copy_uid_model(src, dst)
        self.__copy_attr(src, 'name', dst)
        self.__copy_attr(src, 'description', dst)

    def _copy_project_element_geometry(self, src, dst):
        self.__copy_attr(src, 'origin', dst)
        # do not copy uid-model stuff - date-created and date-modified are taken from content-model.

    def _copy_texture(self, texture_uuid):
        texture_v1 = self.__get_attr(self._project, texture_uuid)
        texture_ = texture.ProjectedTexture()

        self.__require_attr(texture_v1, '__class__', 'ImageTexture')
        self.__copy_attr(texture_v1, 'origin', texture_)
        self.__copy_attr(texture_v1, 'axis_u', texture_)
        self.__copy_attr(texture_v1, 'axis_v', texture_)
        self._copy_content_model(texture_v1, texture_)
        self._copy_image_png(texture_v1, 'image', texture_)
        return texture_

    # textures
    def _copy_textures(self, src, dst):
        texture_uuids = self.__get_attr(src, 'textures')
        dst.textures = [self._copy_texture(texture_uuid) for texture_uuid in texture_uuids]

    # data columns
    def _copy_scalar_data(self, data_v1):
        data = attribute.NumericAttribute()
        self._copy_scalar_array(data_v1, 'array', data)
        if self.__get_attr(data_v1, 'colormap', optional=True) is not None:
            self._test_data_needed(data_v1, data)
        return data

    def _copy_project_element_data(self, data_uuid, valid_locations):
        data_v1 = self.__get_attr(self._project, data_uuid)
        data_class = self.__get_attr(data_v1, '__class__')

        converters = {'ScalarData': self._copy_scalar_data}
        converter = self.__get_attr(converters, data_class)
        data = converter(data_v1)
        location = self.__get_attr(data_v1, 'location')
        if location not in valid_locations:
            raise InvalidOMFFile(f'Invalid data location: {location}')
        self._copy_content_model(data_v1, data)
        return data

    def _copy_data(self, src, dst, valid_locations):
        data_uuids = self.__get_attr(src, 'data', optional=True)
        if data_uuids is None:
            return
        dst.attributes = [self._copy_project_element_data(data_uuid, valid_locations) for data_uuid in data_uuids]

    # points
    def _copy_pointset_element(self, points_v1):
        geometry_uuid = self.__get_attr(points_v1, 'geometry')
        geometry_v1 = self.__get_attr(self._project, geometry_uuid)

        points = pointset.PointSet()
        self._copy_textures(points_v1, points)
        self.__copy_attr(points_v1, 'subtype', points.metadata)
        self._copy_project_element_geometry(geometry_v1, points)
        self._copy_scalar_array(geometry_v1, 'vertices', points)

        valid_locations = ('vertices',)
        return points, valid_locations

    # line sets
    def _copy_lineset_element(self, lines_v1):
        geometry_uuid = self.__get_attr(lines_v1, 'geometry')
        geometry_v1 = self.__get_attr(self._project, geometry_uuid)

        lines = lineset.LineSet()
        self.__copy_attr(lines_v1, 'subtype', lines.metadata)
        self._copy_project_element_geometry(geometry_v1, lines)
        self._copy_scalar_array(geometry_v1, 'vertices', lines)
        self._copy_scalar_array(geometry_v1, 'segments', lines)

        valid_locations = ('vertices', 'segments')
        return lines, valid_locations

    # element list
    def _copy_project_element(self, element_uuid):
        element_v1 = self.__get_attr(self._project, element_uuid)
        element_class = self.__get_attr(element_v1, '__class__')

        converters = {'PointSetElement': self._copy_pointset_element,
                      'LineSetElement': self._copy_lineset_element,
                      'SurfaceElement': self._test_data_needed,
                      'VolumeElement': self._test_data_needed,
                      }
        converter = self.__get_attr(converters, element_class)
        element, valid_locations = converter(element_v1)

        self._copy_content_model(element_v1, element)
        self._copy_data(element_v1, element, valid_locations)
        self.__copy_attr(element_v1, 'color', element.metadata)
        return element

    # main project
    def _copy_project(self, project_uuid):
        project_v1 = self.__get_attr(self._project, project_uuid)
        project = base.Project()

        self._copy_content_model(project_v1, project)
        self.__copy_attr(project_v1, 'author', project.metadata, optional_dst=True, default='')
        self.__copy_attr(project_v1, 'revision', project.metadata, optional_dst=True, default='')
        self.__copy_attr(project_v1, 'date', project.metadata, optional_src=True)
        self.__copy_attr(project_v1, 'units', self._attribute_bucket)  # units have moved to elements.
        self.__copy_attr(project_v1, 'origin', project)

        project.elements = [self._copy_project_element(element) for element in self.__get_attr(project_v1, 'elements')]
        return project
