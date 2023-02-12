"""omf_v1.py: Reader for OMF V1 files."""
import contextlib
import io
import json
import struct
import uuid
import zlib

import numpy as np
import properties

from .interface import IOMFReader, InvalidOMFFile, WrongVersionError

from .. import attribute, base, blockmodel, lineset, pointset, surface, texture

COMPATIBILITY_VERSION = b"OMF-v0.9.0"
_default = object()


# pylint: disable=too-few-public-methods
class Reader(IOMFReader):
    """Reader for OMF V1 files."""

    def __init__(self, filename: str):
        self._filename = filename
        self._f = None
        self._include_binary = True
        self._project = None
        self.__cache = {}  # uuid -> reusable item

    def load(self, include_binary: bool = True, project_json: bool = None):
        self._include_binary = include_binary
        try:
            with open(self._filename, "rb") as self._f:
                project_uuid, json_start = self._read_header()
                self._project = self._read_json(json_start)
                try:
                    return self._convert_project(project_uuid)
                except properties.ValidationError as exc:
                    raise InvalidOMFFile(exc) from exc
        finally:
            self._reset()

    def _reset(self):
        self._f = None
        self._project = None
        self._include_binary = True
        self.__cache = {}  # uuid -> reusable item

    def _read_header(self):
        """Checks magic number and version; gets project uid and json start"""
        self._f.seek(0)
        if self._f.read(4) != b"\x84\x83\x82\x81":
            raise WrongVersionError(f"Unsupported format: {self._filename}")
        file_version = struct.unpack("<32s", self._f.read(32))[0]
        file_version = file_version[0 : len(COMPATIBILITY_VERSION)]
        if file_version != COMPATIBILITY_VERSION:
            raise WrongVersionError("Unsupported file version: {}".format(file_version))
        project_uuid = uuid.UUID(bytes=struct.unpack("<16s", self._f.read(16))[0])
        json_start = struct.unpack("<Q", self._f.read(8))[0]
        return str(project_uuid), json_start

    def _read_json(self, json_start):
        self._f.seek(json_start, 0)
        return json.loads(self._f.read().decode("utf-8"))

    # Safe access to attributes
    @classmethod
    def __get_attr(cls, src, attr, optional=False, default=_default):
        try:
            value = src[attr]
        except TypeError as exc:
            raise InvalidOMFFile(f"Attribute {attr} missing") from exc
        except KeyError as exc:
            if not optional:
                raise InvalidOMFFile(f"Attribute {attr} missing") from exc
            if default is _default:
                return None
            value = default
        return value

    @classmethod
    def __require_attr(cls, src, attr, required_value):
        value = cls.__get_attr(src, attr)
        if value != required_value:
            raise InvalidOMFFile(f"Invalid attribute {attr}. Expected: {required_value}, actual: {value}")

    @classmethod
    # pylint: disable=too-many-arguments
    def __copy_attr(
        cls,
        src,
        src_attr,
        dst,
        dst_attr=None,
        optional_src=False,
        optional_dst=False,
        default=_default,
    ):
        if dst_attr is None:
            dst_attr = src_attr
        try:
            value = src[src_attr]
        except TypeError as exc:
            raise InvalidOMFFile(f"Attribute {src_attr} missing") from exc
        except KeyError as exc:
            if not optional_src:
                raise InvalidOMFFile(f"Attribute {src_attr} missing") from exc
            if default is _default:
                return
            value = default

        if optional_dst and default is not _default and value == default:
            return

        if isinstance(dst, dict):
            dst[dst_attr] = value
        else:
            setattr(dst, dst_attr, value)

    # reading arrays
    @contextlib.contextmanager
    def _override_include_binary(self, temporary_value=True):
        old_value, self._include_binary = self._include_binary, temporary_value
        try:
            yield
        finally:
            self._include_binary = old_value

    def _load_gradient(self, gradient_uuid):
        if gradient_uuid not in self.__cache:
            gradient_v1 = self.__get_attr(self._project, gradient_uuid)
            self.__cache[gradient_uuid] = self._load_array_list(gradient_v1)
        return self.__cache[gradient_uuid]

    def _load_array_list(self, scalar_array):
        return self.__get_attr(scalar_array, "array")

    def _load_array(self, scalar_array):
        scalar_class = self.__get_attr(scalar_array, "__class__")
        converters = {
            "StringArray": self._load_array_list,
            "DateTimeArray": self._load_array_list,
            "ColorArray": self._load_array_list,
        }
        shape_lookup = {
            "ScalarArray": ("*",),
            "Int2Array": ("*", 2),
            "Int3Array": ("*", 3),
            "Vector2Array": ("*", 2),
            "Vector3Array": ("*", 3),
        }
        converter = self.__get_attr(converters, scalar_class, optional=True)
        if converter is not None:
            return converter(scalar_array)

        shape = self.__get_attr(shape_lookup, scalar_class)
        shape = tuple(-1 if s == "*" else s for s in shape)
        base_vector = self.__get_attr(scalar_array, "array")

        start = self.__get_attr(base_vector, "start")
        length = self.__get_attr(base_vector, "length")
        dtype = self.__get_attr(base_vector, "dtype")

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

    def _convert_image(self, image_png):
        start = self.__get_attr(image_png, "start")
        length = self.__get_attr(image_png, "length")
        self.__require_attr(image_png, "dtype", "image/png")
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
        image = self._convert_image(image_png)
        setattr(dst, dst_attr, image)

    # base-class handlers
    def _copy_uid_model(self, src, dst):
        self.__copy_attr(src, "date_created", dst.metadata)
        self.__copy_attr(src, "date_modified", dst.metadata)

    def _copy_content_model(self, src, dst):
        self._copy_uid_model(src, dst)
        self.__copy_attr(src, "name", dst)
        self.__copy_attr(src, "description", dst)

    def _copy_project_element_geometry(self, src, dst):
        self.__copy_attr(src, "origin", dst)
        # do not copy uid-model stuff - date-created and date-modified are taken from content-model.

    def _convert_texture(self, texture_uuid):
        texture_v1 = self.__get_attr(self._project, texture_uuid)
        texture_ = texture.ProjectedTexture()

        self.__require_attr(texture_v1, "__class__", "ImageTexture")
        self.__copy_attr(texture_v1, "origin", texture_)
        self.__copy_attr(texture_v1, "axis_u", texture_)
        self.__copy_attr(texture_v1, "axis_v", texture_)
        self._copy_content_model(texture_v1, texture_)
        self._copy_image_png(texture_v1, "image", texture_)
        return texture_

    # textures
    def _copy_textures(self, src, dst):
        texture_uuids = self.__get_attr(src, "textures", optional=True, default=[])
        dst.textures = [self._convert_texture(texture_uuid) for texture_uuid in texture_uuids]

    # data columns
    def _convert_colormap(self, colormap_uuid):
        colormap_v1 = self.__get_attr(self._project, colormap_uuid)
        self.__require_attr(colormap_v1, "__class__", "ScalarColormap")
        gradient_uuid = self.__get_attr(colormap_v1, "gradient")

        colormap = attribute.ContinuousColormap()
        colormap.gradient = self._load_gradient(gradient_uuid)
        self.__copy_attr(colormap_v1, "limits", colormap)
        self._copy_content_model(colormap_v1, colormap)
        return colormap

    def _convert_scalar_data(self, data_v1):
        data = attribute.NumericAttribute()
        self._copy_scalar_array(data_v1, "array", data)
        colormap_uuid = self.__get_attr(data_v1, "colormap", optional=True)
        if colormap_uuid is not None:
            data.colormap = self._convert_colormap(colormap_uuid)
        return [data]

    def _convert_vector_data(self, data_v1):
        data = attribute.VectorAttribute()
        self._copy_scalar_array(data_v1, "array", data)
        return [data]

    def _convert_string_data(self, data_v1):
        data = attribute.StringAttribute()
        self._copy_scalar_array(data_v1, "array", data)
        return [data]

    def _mapped_column_to_category(self, legend_v1, data_v1, data_column, color_column):
        colormap = attribute.CategoryColormap()

        length = len(data_column)
        colormap.indices = list(range(length))
        colormap.values = data_column
        if color_column is not None:
            colormap.colors = color_column
        self._copy_content_model(legend_v1, colormap)

        catgory_attribute = attribute.CategoryAttribute()
        self._copy_scalar_array(data_v1, "array", catgory_attribute)
        catgory_attribute.categories = colormap

        return catgory_attribute

    def _convert_mapped_data(self, data_v1):
        # This is messy because of changes from V1 to V2.
        # V1's MappedData contains an index array and an arbitrary list of Legends
        # V2's CategoryAttribute contains the same index array but only allows one CategoryColormap.
        # The CategoryColormap can hold one value column (strings) and optionally one color column.
        # To complicate things further, V1 did not enforce that all Legends in MappedData have the same length.
        # This function attempts to preserve data/color groups:
        # - it splits the Legends into colors and other columns
        # - for each 'other' column, it creates a CategoryAttribute with a CategoryColormap
        # - CategoryColormap.indices will be filled with a default range (0..N) because this did not exist in V1
        # - it takes the first available color column with the same length and adds it to that CategoryColormap
        # - if no matching color column is available, the color attribute of the CategoryColormap remains unset
        # - any leftover color columns will be turned into data columns - rendering them effectively useless since
        #   data columns only support string data, e.g. these will turn into "255,255,255" for a white color.

        # Step 1: load all legends and split them into 'color' and 'other' columns.
        color_columns, other_columns = [], []

        for legend_uuid in self.__get_attr(data_v1, "legends"):
            legend_v1 = self.__get_attr(self._project, legend_uuid)
            self.__require_attr(legend_v1, "__class__", "Legend")

            values_uuid = self.__get_attr(legend_v1, "values")
            array_v1 = self.__get_attr(self._project, values_uuid)
            array_class = self.__get_attr(array_v1, "__class__")

            with self._override_include_binary(temporary_value=True):
                column = self._load_array(array_v1)

            if array_class == "ColorArray":
                color_columns.append((legend_v1, column))
            else:
                other_columns.append((legend_v1, column, array_class))

        # process 'other' columns and add a color column if possible
        for legend_v1, column, array_class in other_columns:
            length = len(column)

            # find a matching color column
            color_column = None
            matching_color_column = next(((lgd, col) for lgd, col in color_columns if len(col) == length), None)
            if matching_color_column is not None:
                color_columns.remove(matching_color_column)
                _, color_column = matching_color_column

            yield self._mapped_column_to_category(legend_v1, data_v1, column, color_column)

        # process remaining color columns
        for legend_v1, color_column in color_columns:
            # convert color to text but preserve the color column - this gives pretty colored colors.
            column = [",".join(map(str, color)) for color in color_column]
            yield self._mapped_column_to_category(legend_v1, data_v1, column, color_column)

    def _copy_project_element_data(self, data_uuid, valid_locations, attributes):
        data_v1 = self.__get_attr(self._project, data_uuid)
        data_class = self.__get_attr(data_v1, "__class__")

        converters = {
            "ColorData": self._convert_vector_data,
            "DateTimeData": self._convert_string_data,
            "MappedData": self._convert_mapped_data,
            "ScalarData": self._convert_scalar_data,
            "StringData": self._convert_string_data,
            "Vector2Data": self._convert_vector_data,
            "Vector3Data": self._convert_vector_data,
        }
        converter = self.__get_attr(converters, data_class)
        location = self.__get_attr(data_v1, "location")
        if location not in valid_locations:
            raise InvalidOMFFile(f"Invalid data location: {location}")
        for data in converter(data_v1):
            data.location = location
            self._copy_content_model(data_v1, data)
            attributes.append(data)

    def _copy_data(self, src, dst, valid_locations):
        data_uuids = self.__get_attr(src, "data", optional=True, default=[])
        for data_uuid in data_uuids:
            self._copy_project_element_data(data_uuid, valid_locations, dst.attributes)

    # points
    def _convert_pointset_element(self, points_v1):
        geometry_uuid = self.__get_attr(points_v1, "geometry")
        geometry_v1 = self.__get_attr(self._project, geometry_uuid)
        self.__require_attr(geometry_v1, "__class__", "PointSetGeometry")

        points = pointset.PointSet()
        self._copy_textures(points_v1, points)
        self.__copy_attr(points_v1, "subtype", points.metadata)
        self._copy_project_element_geometry(geometry_v1, points)
        self._copy_scalar_array(geometry_v1, "vertices", points)

        valid_locations = ("vertices",)
        return points, valid_locations

    # line sets
    def _convert_lineset_element(self, lines_v1):
        geometry_uuid = self.__get_attr(lines_v1, "geometry")
        geometry_v1 = self.__get_attr(self._project, geometry_uuid)
        self.__require_attr(geometry_v1, "__class__", "LineSetGeometry")

        lines = lineset.LineSet()
        self.__copy_attr(lines_v1, "subtype", lines.metadata)
        self._copy_project_element_geometry(geometry_v1, lines)
        self._copy_scalar_array(geometry_v1, "vertices", lines)
        self._copy_scalar_array(geometry_v1, "segments", lines)

        valid_locations = ("vertices", "segments")
        return lines, valid_locations

    # surfaces - triangulated or gridded
    def _convert_surface_geometry(self, geometry_v1):
        surface_ = surface.Surface()
        self._copy_project_element_geometry(geometry_v1, surface_)
        self._copy_scalar_array(geometry_v1, "vertices", surface_)
        self._copy_scalar_array(geometry_v1, "triangles", surface_)

        valid_locations = ("vertices", "faces")
        return surface_, valid_locations

    def _convert_surface_grid_geometry(self, geometry_v1):
        surface_ = surface.TensorGridSurface()
        self._copy_project_element_geometry(geometry_v1, surface_)
        self.__copy_attr(geometry_v1, "tensor_u", surface_)
        self.__copy_attr(geometry_v1, "tensor_v", surface_)
        self.__copy_attr(geometry_v1, "axis_u", surface_)
        self.__copy_attr(geometry_v1, "axis_v", surface_)

        valid_locations = ("vertices", "faces")
        return surface_, valid_locations

    def _convert_surface_element(self, surface_v1):
        geometry_uuid = self.__get_attr(surface_v1, "geometry")
        geometry_v1 = self.__get_attr(self._project, geometry_uuid)
        geometry_class = self.__get_attr(geometry_v1, "__class__")
        converters = {
            "SurfaceGeometry": self._convert_surface_geometry,
            "SurfaceGridGeometry": self._convert_surface_grid_geometry,
        }
        converter = self.__get_attr(converters, geometry_class)
        surface_, valid_locations = converter(geometry_v1)
        self.__copy_attr(surface_v1, "subtype", surface_.metadata)
        self._copy_textures(surface_v1, surface_)

        return surface_, valid_locations

    # volumes
    def _convert_volume_element(self, volume_v1):
        geometry_uuid = self.__get_attr(volume_v1, "geometry")
        geometry_v1 = self.__get_attr(self._project, geometry_uuid)
        self.__require_attr(geometry_v1, "__class__", "VolumeGridGeometry")
        volume = blockmodel.TensorGridBlockModel()
        self.__copy_attr(volume_v1, "subtype", volume.metadata)
        self._copy_project_element_geometry(geometry_v1, volume)
        self.__copy_attr(geometry_v1, "tensor_u", volume)
        self.__copy_attr(geometry_v1, "tensor_v", volume)
        self.__copy_attr(geometry_v1, "tensor_w", volume)
        self.__copy_attr(geometry_v1, "axis_u", volume)
        self.__copy_attr(geometry_v1, "axis_v", volume)
        self.__copy_attr(geometry_v1, "axis_w", volume)

        valid_locations = ("vertices", "cells")
        return volume, valid_locations

    # element list
    def _convert_project_element(self, element_uuid):
        element_v1 = self.__get_attr(self._project, element_uuid)
        element_class = self.__get_attr(element_v1, "__class__")

        converters = {
            "PointSetElement": self._convert_pointset_element,
            "LineSetElement": self._convert_lineset_element,
            "SurfaceElement": self._convert_surface_element,
            "VolumeElement": self._convert_volume_element,
        }
        converter = self.__get_attr(converters, element_class)
        element, valid_locations = converter(element_v1)

        self._copy_content_model(element_v1, element)
        self._copy_data(element_v1, element, valid_locations)
        self.__copy_attr(element_v1, "color", element.metadata)
        return element

    # main project
    def _convert_project(self, project_uuid):
        project_v1 = self.__get_attr(self._project, project_uuid)
        project = base.Project()

        self._copy_content_model(project_v1, project)
        self.__copy_attr(project_v1, "author", project.metadata, optional_dst=True, default="")
        self.__copy_attr(project_v1, "revision", project.metadata, optional_dst=True, default="")
        self.__copy_attr(project_v1, "date", project.metadata, optional_src=True)
        self.__copy_attr(
            project_v1,
            "units",
            project.metadata,
            "spacial_units",
            optional_dst=True,
            default="",
        )
        self.__copy_attr(project_v1, "origin", project)

        project.elements = [
            self._convert_project_element(element) for element in self.__get_attr(project_v1, "elements")
        ]
        return project
