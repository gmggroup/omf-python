"""omf: API library for Open Mining Format file interchange format"""

from .base import Project
from .data import (
    ColorArray,
    ColorData,
    DateTimeArray,
    DateTimeColormap,
    DateTimeData,
    Legend,
    MappedData,
    ScalarArray,
    ScalarColormap,
    ScalarData,
    StringArray,
    StringData,
    Vector2Array,
    Vector2Data,
    Vector3Array,
    Vector3Data,
)
from .fileio import GeoH5Writer, OMFReader, OMFWriter
from .lineset import LineSetElement, LineSetGeometry
from .pointset import PointSetElement, PointSetGeometry
from .surface import SurfaceElement, SurfaceGeometry, SurfaceGridGeometry
from .texture import ImageTexture
from .volume import VolumeElement, VolumeGridGeometry

__version__ = "3.0.0-alpha"
__author__ = "Global Mining Standards and Guidelines Group"
__license__ = "MIT License"
__copyright__ = "Copyright 2017 Global Mining Standards and Guidelines Group"
